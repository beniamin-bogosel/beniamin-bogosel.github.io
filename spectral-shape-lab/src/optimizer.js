import {compileExpression, DomainError} from './expression.js';
import {amplitudes, coefficients, dot, norm, project} from './geometry.js';

export function validateSettings(settings) {
  const s={expression:settings.expression, area:Number(settings.area), starts:Number(settings.starts),
    steps:Number(settings.steps), seed:Number(settings.seed), maximize:Boolean(settings.maximize)};
  if(!Number.isFinite(s.area)||s.area<1e-8||s.area>1e8)throw Error('Area must be between 10⁻⁸ and 10⁸.');
  if(!Number.isInteger(s.starts)||s.starts<1||s.starts>32)throw Error('Choose 1–32 starts.');
  if(!Number.isInteger(s.steps)||s.steps<0||s.steps>2000)throw Error('Choose 0–2000 steps per start.');
  if(!Number.isInteger(s.seed)||s.seed<0||s.seed>0xffffffff)throw Error('Seed must be an integer between 0 and 4294967295.');
  compileExpression(s.expression); return s;
}
export function randomGenerator(seed) {
  let a=seed>>>0;
  return ()=>{a=(a+0x6D2B79F5)>>>0;let t=a;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};
}
export function objectiveEvaluator(model, functional, area, scale=1, maximize=false) {
  const sign=maximize?-1:1;
  return q=>{
    const record=model.forward(q), spectrum=record.mean.map(v=>v/area);
    const {value:F,partials}=functional.evaluate(spectrum);
    const gradient=model.vjp(record,partials.map(v=>sign*v/(scale*area)));
    return {F,objective:sign*F/scale,gradient,spectrum,J:record.mean,std:record.std.map(v=>v/area),partials};
  };
}
export async function chooseStarts(context, settings, functional, checkpoint=async()=>{}) {
  const {J,q,dimension:d}=context, scores=[], sign=settings.maximize?-1:1;let skipped=0;
  for(let i=0;i<J.length/10;i++) {
    if(i%500===0)await checkpoint();
    try {scores.push({id:i,value:sign*functional.evaluate(J.subarray(i*10,i*10+10).map(v=>v/settings.area)).value});}
    catch(error){if(!(error instanceof DomainError))throw error;skipped++;}
  }
  if(scores.length<settings.starts-1)throw Error('Too few training starts in the domain of this function.');
  scores.sort((a,b)=>a.value-b.value||a.id-b.id);
  const count=Math.min(Math.floor(settings.starts/2),settings.starts-1),best=scores.slice(0,count), pool=scores.slice(count).sort((a,b)=>a.id-b.id);
  const random=randomGenerator(settings.seed), selected=[];
  for(let i=0;i<settings.starts-1-count;i++) {const j=i+Math.floor(random()*(pool.length-i));[pool[i],pool[j]]=[pool[j],pool[i]];selected.push(pool[i]);}
  const point=(r,kind)=>({source:`${kind}_${r.id}`,trainingRow:r.id,q:Float64Array.from(q.subarray(r.id*d,(r.id+1)*d))});
  return {starts:[{source:'disk',trainingRow:null,q:new Float64Array(d)},...best.map(r=>point(r,'training')),...selected.map(r=>point(r,'random'))],skipped};
}

export async function optimize(context, model, support, input, hooks={}) {
  const settings=validateSettings(input), functional=compileExpression(settings.expression);
  const {rho,caps,id,name}=context.bundle, d=context.dimension, m=context.modes;
  const diskValue=objectiveEvaluator(model,functional,settings.area)(new Float64Array(d)).F;
  const scale=Math.max(1,Math.abs(diskValue)),evaluate=objectiveEvaluator(model,functional,settings.area,scale,settings.maximize);
  const now=()=>performance.now(), begin=now(), checkpoint=hooks.checkpoint||(()=>Promise.resolve(true));
  const projectShape=q=>project(q,rho,caps);
  const median=[...support.scale.slice(0,3)].sort((a,b)=>a-b)[Math.min(1,m-1)];
  const preconditioner=Float64Array.from(support.scale,v=>m>3?(v/median)**2:1);
  let best=null,stopped=false,events=0,lastEmission=-Infinity; const runs=[],trace=[],skippedStarts=[];
  hooks.phase?.('Ranking training starts');
  const selection=await chooseStarts(context,settings,functional,checkpoint);
  const candidate=(q,value,source,kind,iterations,termination,pg)=>({q:Array.from(q),coefficients:coefficients(q,settings.area),
    F:value.F,objective:value.objective,predictedEigenvalues:Array.from(value.spectrum),predictedJ:Array.from(value.J),
    ensembleStd:Array.from(value.std),functionalPartials:Array.from(value.partials),source,kind,iterations,termination,
    projectedGradientNorm:pg,amplitudeSum:amplitudes(q).reduce((s,v)=>s+v,0)});
  const emit=(current,start,iteration,force=false)=>{
    if(!best||current.objective<best.objective)best=current;
    trace.push({event:events++,start,iteration,F:current.F,bestF:best.F});
    if(force||now()-lastEmission>100) {hooks.progress?.({current,best,start,iteration,trace:trace.slice(-1),elapsedSeconds:(now()-begin)/1000});lastEmission=now();}
  };
  hooks.phase?.('Optimizing');
  for(let s=0;s<selection.starts.length;s++) {
    if(await checkpoint()===false){stopped=true;break;}
    const start=selection.starts[s];let q=projectShape(start.q),value;
    try {value=evaluate(q);}catch(error){if(!(error instanceof DomainError))throw error;skippedStarts.push(start.source);continue;}
    if(!support.contains(q))throw Error('An initial shape lies outside the saved training support.');
    const initialProjection=projectShape(q.map((v,j)=>v-value.gradient[j]));
    let pg=norm(q.map((v,j)=>v-initialProjection[j]));
    const incumbent=candidate(q,value,start.source,'incumbent',0,'incumbent',pg);
    incumbent.startQ=Array.from(start.q);runs.push(incumbent);emit(incumbent,s,0,true);
    let step=.2,iterations=0,termination=settings.steps?'iteration_budget':'incumbent';const history=[];
    for(let it=0;it<settings.steps;it++) {
      if(await checkpoint()===false){stopped=true;termination='stopped';break;}
      const projected=projectShape(q.map((v,j)=>v-value.gradient[j]));pg=norm(q.map((v,j)=>v-projected[j]));
      history.push({iteration:it,F:value.F,objective:value.objective,projectedGradientNorm:pg});iterations++;
      if(pg<2e-5){termination='projected_gradient';break;}
      let accepted=false;
      for(let trial=0;trial<18;trial++) {
        if(await checkpoint()===false){stopped=true;termination='stopped';break;}
        let update=value.gradient.map((v,j)=>step*v*preconditioner[j]);const factor=Math.min(1,.06/Math.max(norm(update),1e-20));update=update.map(v=>v*factor);
        const next=project(q.map((v,j)=>v-update[j]),rho,caps,preconditioner.subarray(0,m));
        if(!support.contains(next)){step*=.5;continue;}
        let trialValue;
        try {trialValue=evaluate(next);}catch(error){if(!(error instanceof DomainError))throw error;step*=.5;continue;}
        if(trialValue.objective<=value.objective+1e-4*dot(value.gradient,next.map((v,j)=>v-q[j]))) {
          q=next;value=trialValue;step=Math.min(step*1.3,3);accepted=true;
          emit(candidate(q,value,start.source,'gradient_run',iterations,'running',pg),s,it+1);break;
        }
        step*=.5;
      }
      if(stopped)break;
      if(!accepted||step<1e-8){termination=accepted?'step_too_small':'line_search_or_support';break;}
    }
    const projected=projectShape(q.map((v,j)=>v-value.gradient[j]));pg=norm(q.map((v,j)=>v-projected[j]));
    const endpoint=candidate(q,value,start.source,'gradient_run',iterations,termination,pg);
    endpoint.history=history;endpoint.startQ=Array.from(start.q);runs.push(endpoint);emit(endpoint,s,iterations,true);
    if(stopped)break;
  }
  // Pick a persisted endpoint/incumbent, rather than a transient progress record.
  best=runs.reduce((win,r)=>!win||r.objective<win.objective?r:win,null);
  if(best) {
    best.supportDistance=support.distance(best.q);
    if(best.amplitudeSum>rho+1e-10||!support.contains(best.q))throw Error('Final shape failed its feasibility check.');
  }
  return {format:'spectral-shape-run',version:1,appVersion:'1.0.0',createdAt:new Date().toISOString(),
    status:stopped?'stopped':'complete',selection:'surrogate only; no PDE solve',
    model:{id,name,bundleSHA256:context.bundleSHA256,source:context.source},settings,
    objectiveScale:scale,symbolicGradient:functional.derivatives,area:settings.area,rho,caps,
    relativeRadialFloor:1-rho,supportRadius:support.radius,supportRoundoffMargin:support.margin,
    startRng:'mulberry32 (browser protocol)',invalidTrainingRows:selection.skipped,skippedStarts,
    starts:selection.starts.map(s=>({...s,q:Array.from(s.q)})),best,runs,trace,elapsedSeconds:(now()-begin)/1000};
}

/** Exact nearest-neighbour search over the full finite symmetry orbit, without
 * storing 48 copies. Equal pair scales make rotations/reflections isometries.
 */
import {rotate} from './geometry.js';

export class TrainingSupport {
  constructor(context) {
    this.dimension=context.dimension;this.scale=context.bundle.support.scale;
    this.radius=context.bundle.support.radius;this.margin=context.bundle.support.roundoffBound;
    this.count=context.bundle.support.count;
    const d=this.dimension;
    this.points=new Float64Array((this.count+1)*d);
    for(let i=0;i<context.q.length;i++) this.points[i]=context.q[i]/this.scale[i%d];
    const ids=Array.from({length:this.count+1},(_,i)=>i);
    this.nodes=[];
    const build=(indices,depth=0)=>{
      if(!indices.length)return -1;
      // Maximum-spread split performs better than cycling through high harmonics.
      let axis=0,best=-1;
      for(let j=0;j<d;j++) {
        let lo=Infinity,hi=-Infinity;
        for(const id of indices){const v=this.points[id*d+j];if(v<lo)lo=v;if(v>hi)hi=v;}
        if(hi-lo>best){best=hi-lo;axis=j;}
      }
      indices.sort((a,b)=>this.points[a*d+axis]-this.points[b*d+axis]);
      const mid=Math.floor(indices.length/2), index=this.nodes.length;
      const node={id:indices[mid],axis,left:-1,right:-1};this.nodes.push(node);
      node.left=build(indices.slice(0,mid),depth+1);node.right=build(indices.slice(mid+1),depth+1);
      return index;
    };
    this.root=build(ids);
  }
  nearestBase(point, upper=Infinity, firstWithin=false) {
    let best=upper,found=false; const d=this.dimension,points=this.points;
    const visit=index=>{
      if(index<0 || (firstWithin&&found))return;
      const node=this.nodes[index], offset=node.id*d;let dist=0;
      for(let j=0;j<d;j++){dist+=(point[j]-points[offset+j])**2;if(dist>best)break;}
      if(dist<=best){best=dist;found=true;}
      const delta=point[node.axis]-points[offset+node.axis];
      visit(delta<0?node.left:node.right);
      if(delta*delta<=best)visit(delta<0?node.right:node.left);
    };
    visit(this.root);return {squared:best,found};
  }
  query(q, membershipOnly=false) {
    // The dihedral orbit is closed under inversion, so querying transformed q
    // against base samples is equivalent to querying q against transformed samples.
    const normalized=Float64Array.from(q,(v,j)=>v/this.scale[j]);
    const threshold=(this.radius-this.margin)**2;
    let best=membershipOnly?threshold:Infinity;
    for(let i=0;i<24;i++) for(const reflected of [false,true]) {
      const p=rotate(normalized,2*Math.PI*i/24,reflected);
      const result=this.nearestBase(p,best,membershipOnly);
      if(membershipOnly&&result.found)return true;
      if(result.found)best=result.squared;
    }
    return membershipOnly?false:Math.sqrt(best);
  }
  contains(q) {return this.query(q,true);}
  distance(q) {return this.query(q,false);}
}

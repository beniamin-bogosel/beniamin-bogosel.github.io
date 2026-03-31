package test;

public class Triangle {
	//Atribute
	public double a;
	public double b;
	public double c;
	
	public double ha;
	public double hb;
	public double hc;
	
	
	//Constructor
	public Triangle(double aa, double bb, double cc) {
		// Conditii? 
		// numere pozitive?
		// a < b+ c si celelalte variante?
		
		this.a = aa;
		this.b = bb;
		this.c = cc;
	}
	
	// Metoda
	// Aria cu Heron
	public double Area(double a, double b, double c) {
		double p;
		p = (a+b+c)/2;
		double S;
		S = Math.sqrt(p*(p-a)*(p-b)*(p-c));
		return S;
	}
	public double Area(double a, double ha) {
		return (a*ha)/2;
	}
	
	public void ComputeHeights() {
		this.ha = 2*this.Area(this.a,this.b,this.c)/this.a;
		this.hb = 2*this.Area(this.a,this.b,this.c)/this.b;
		this.hc = 2*this.Area(this.a,this.b,this.c)/this.c;
	}
	
	
}

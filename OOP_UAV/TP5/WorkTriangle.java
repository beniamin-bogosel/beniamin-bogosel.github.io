package test;

public class WorkTriangle {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		Triangle Tri1 = new Triangle(1,2,5);
		
		Tri1.ComputeHeights();
		
		System.out.println("Aria cu Heron = "+Tri1.Area(Tri1.a, Tri1.b, Tri1.c));
		
		System.out.println("Aria cu b*h/2 = "+Tri1.Area(Tri1.a, Tri1.ha));

	}

}

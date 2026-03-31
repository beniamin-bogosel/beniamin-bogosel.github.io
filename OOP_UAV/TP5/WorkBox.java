package test;

public class WorkBox {
	
	public static double add(double a, double b) {
		return a+b;
	}
	
	public static double Dubleaza(double a) {
		a = 2*a;
		return a;
	}
	
	public static void Change(Box b) {
		
		b.open();
		System.out.println("In functie = "+b.closed());
	}
	
	

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		Box Cutia1 = new Box("Un alt nume frumos",2);

		System.out.println("Continut = "+Cutia1.contents());	
		Cutia1.clear("Parola Incorecta");
		
		//Thief.steal(Cutia1);
		Cutia1.clear(Cutia1.getPass());
		System.out.println("Continut dupa = "+Cutia1.contents());	

		System.out.println(add(3.5,9.2));
		
		double a = 2;
		System.out.println(Dubleaza(a));
		
		System.out.println(a);
		
		Cutia1.close();
		System.out.println("Inainte = "+Cutia1.closed());

		
		Change(Cutia1);

		System.out.println("Dupa = "+Cutia1.closed());

		
	}

}

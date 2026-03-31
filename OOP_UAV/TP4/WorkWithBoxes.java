package test;

public class WorkWithBoxes {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		// Definim un obiect Box
		
		Box Cutia1 = new Box("Un alt nume frumos",2);
		
		
		//Cutia1.name = "Altceva";
		Cutia1.ChangeName("Altceva");
		
		System.out.println(Cutia1.name());
		System.out.println("Continut 1: "+Cutia1.contents());
		
		Box Cutia2 = new Box(3);
		
		System.out.println(Cutia2.name());
		System.out.println("Continut 2: "+Cutia2.contents());

		System.out.println("C1 inchisa? "+Cutia1.State());
		
		System.out.println("Continut C1 inainte= " + Cutia1.contents());
		Cutia1.open();

		Cutia1.fill(10);
		System.out.println("Continut C1 dupa     " + Cutia1.contents());

		System.out.println("C1 inchisa? "+Cutia1.State());

		
		
		
	}

}

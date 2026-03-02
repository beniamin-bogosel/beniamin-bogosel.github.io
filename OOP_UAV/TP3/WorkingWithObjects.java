package sesiunea3;

public class WorkingWithObjects {

	public static void main(String[] args) {
		// Definim obiecte de tip Box, din clasa Box

		Box b1 = new Box("Cutie",2);
		Box b2 = new Box("Cutie mai buna",2);

		
		
		System.out.println(b1.name());
		System.out.println(b2.name());
		
		Box b3 = new Box(20);
		
		System.out.println(b3.name());
		
		System.out.println(b3.continut());
		//b3.name()  = "Cutie nestandard";
		
		System.out.println(b3.name());

		System.out.println("Cutia e inchisa: "+b3.closed());
		b3.open();
		System.out.println("Cutia e inchisa: "+b3.closed());
		b3.close();
		System.out.println("Cutia e inchisa: "+b3.closed());

		System.out.println("Continutul lui b3: "+b3.continut());
		b3.open();
		System.out.println("b3 inchisa?      : "+b3.closed());

		b3.fill(10);
		System.out.println("Continutul lui b3: "+b3.continut());

	}

}

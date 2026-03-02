package sesiunea3;

public class TestAnBisect {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		int Y = 2100;
		
		
		// Y nu e multiplu de 100 si e multiplu de 4 -> an bisect
		// Y e divizibil cu 400 -> an bisect
		
		boolean bisect=false;
		if (Y%100!=0 && Y%4==0)
			bisect = true;
		if (Y%400==0)
			bisect = true;
		
		if (bisect) {
			System.out.println("An bisect");
		}
		else
			System.out.println("Anul nu este bisect");
			
	}
	

}

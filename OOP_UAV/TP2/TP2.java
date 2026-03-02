package curs2;

public class TP2 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int i = 0;
		System.out.println(i);
		
		double fahr ;
		double celsius ;
		
		fahr = 100;
		celsius = 5/9 * ( fahr - 32) ;
		
		
		System.out.println("Fahrenheit = "+fahr);
		System.out.println("Celsius    = "+celsius);
		
		// 5/9 ca si tip int
		System.out.println(5/9);
		
		// 5/9 ca si double
		// Daca rezultatul calculului este real, nu folosim
		// numere "intregi". 
		System.out.println(5.0/9.0);

		celsius = 5.0/9.0 * (fahr - 32.0) ;
		
		
		System.out.println("Fahrenheit = "+fahr);
		System.out.println("Celsius    = "+celsius);
		
		// constantele se definesc cu "final"
		final int YEAR = 1984;
		
		System.out.println(YEAR);

		double pi = 3.1415;
		double val = Math.sin(pi);
		System.out.println("Sin(pi)    = "+val);

		double pp = Math.pow(3,5);
		System.out.println(pp);

		//Math.exp(2.34);
	}

}

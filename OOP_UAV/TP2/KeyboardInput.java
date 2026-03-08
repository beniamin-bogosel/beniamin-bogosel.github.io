package test;

import java.util.*;

public class KeyboardInput {
	
	static final Scanner input = new Scanner(System.in);

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int    intreg;
		double real;
		String c; // definirea unui sir de caractere
	
		//System.out.println("Introducet un nr. intreg: ");
		//intreg = input.nextInt();
		//System.out.println("Nr introdus este: "+intreg);

		//System.out.println("Introducet un nr. intreg urmat de un nr. real: ");
		//intreg = input.nextInt();
		//real   = input.nextDouble();
		
		//System.out.println("Intreg = "+intreg+"  Real="+real);
		
		// Sir de caractere
		System.out.println("Introduceti un sir de caractere: ");
		c = input.next(); // Se citeste urmatorul cuvant! 
		
		System.out.println("S-a introdus: "+c);
		
	}

}

package curs1;

import java . util .*;

public class PrimaClasa {
	
	static final Scanner input = new Scanner ( System . in ) ;

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		// print: cursorul ramane pe aceeasi linie
		// println: cursorul se muta pe linia urmatoare
		System.out.println("Hello\n \t World" + " " + "sir ce urmeaza dupa");
		
		// Variabila String
		String SirDeCaractere;
		SirDeCaractere="Hello World!";
		String SirDeCaractere2=" Un alt String!";

		System.out.println(SirDeCaractere+SirDeCaractere2);
		
		// Pentru siruri de caractere + inseamna concatenare
		
		char varA; // ghilimele simple pentru caractere
		           // ghilimele duble pentru stringuri
		varA = '&';
		
		System.out.println(varA);
		
		byte bb=127;
		System.out.println(bb);
		
		// Variabila literala = scriere explicita cu caractere
		// 12345789325234 variabila literala - de tip int
		// pentru a fi considerata long, trebuie sa adaugam L
		
		long i=12345123451L;
		System.out.println(i);
		
		boolean bool=true; 
		System.out.println(bool);

		byte b = 1;
		b = (byte)(b + b);
		
		System.out.println(b);

		// literal tip real -> double
		// adaugam f
		float f1=1.1234567f, f2=0.00000001f;
		
		// pierdere de precizie lucrand cu numere in precizie flotanta
		System.out.println(f1);
		System.out.println(f1+f2);

		// Copy-paste din fisierul Exercitii TP
		int intreg = 2014;
		double real = 3.1416;
		char litera = 'S';
		System . out . println ( intreg ) ;
		// Conversie automata cand utilizam "+" 
		System . out . println (" intreg : " + intreg ) ;
		System . out . println (" real ..: " + real ) ;
		System . out . println (" litera : " + litera ) ;

		double fahr ;
		double celsius ;
		
		System.out.println("Calcul "+ (5.0 / 2));
		
		System . out . println (" Program de conversie deg . F in deg . C ") ;
		System . out . println (" Introduceti o temperatura : ") ;
		
		fahr = input . nextDouble () ;
		
		System.out.println(fahr);
		
		celsius = 5 * ( fahr - 32) / 9;
		celsius = 5 / 9.0 * ( fahr - 32) ;
		
		// Atentie la tipul operatiilor, la tipul variabilelor
		
		
		System . out . println (" Temperatura in Celsius este : " + celsius);
	}

}

package sesiunea3;

import java.util.Scanner;

public class TestIf {
	
	static final Scanner input = new Scanner ( System . in ) ;

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		int x , y ;
		// ...
		System.out.print("Introduceti primul numar: ");
		x = input . nextInt () ;
		
		System.out.print("Introduceti al doilea numar: ");
		y = input . nextInt () ;
		// Blocurile ce contin o singura instructiune nu necesita {... } 
		if ( x > y ) {
			System . out . println (" Primul nr. e mai mare ") ;
			System . out . println (" Primul nr. e mai mare ") ;
		}
		else {
			System . out . println (" Al doilea nr. e mai mare ") ;
		}
		
	}

}

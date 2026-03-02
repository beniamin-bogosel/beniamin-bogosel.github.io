package sesiunea3;

import java.util.Scanner;

public class BooleanTypes {
	
	static final Scanner input = new Scanner ( System . in ) ;
	
	public static void main(String[] args) {
		
		boolean var;
		var = true;
		
		System.out.println(var);
		
		int x ;
		System.out.print("Introduceti un nr intreg: ");

		//x = input . nextInt () ;
		//boolean b = x > 0 && x < 10;
		//System.out.println("Valoarea expresiei booleene este: "+b);
		
		x = input . nextInt () ;

		//boolean b = x > 0 && Math.sqrt(x) < 10;
		boolean b =  Math.sqrt(x) < 10 && x<0;

		System.out.println("Valoarea expresiei booleene este: "+!b);
	}

}

package sesiunea3;

import java.util.Scanner;

public class EcuatieGradul2 {
	
	static final Scanner input = new Scanner ( System . in ) ;

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		double a, b, c;
		
		System.out.print("a=");
		a = input . nextDouble () ;
		
		System.out.print("b=");
		b = input . nextDouble () ;
		
		System.out.print("c=");
		c = input . nextDouble () ;
		
		System.out.println(a);
		System.out.println(b);
		System.out.println(c);
		
		double delta = b*b-4*a*c;
		
		System.out.println("Delta = "+delta);

		if (a!=0) {
			if (delta==0) {
				System.out.println("O singura solutie: x="+(-b/(2*a)));
			}
			else {
				if (delta>0) {
					System.out.println("Doua radacini reale");
					double sq = Math.sqrt(delta);
					System.out.println("x1 = "+(-b+sq)/(2*a));
					System.out.println("x2 = "+(-b-sq)/(2*a));
	
				} 
				else {
					System.out.println("Doua radacini complexe");
					double sq = Math.sqrt(-delta);
					System.out.println("x1 = "+(-b)/(2*a)+"+"+sq/(2*a)+"i");
					System.out.println("x2 = "+(-b)/(2*a)+"-"+sq/(2*a)+"i");
				}
				
			}
		}
		else {
			if(b!=0) {
				System.out.println("a=0: ecuatie de gradul 1");
				System.out.println("Solutia = "+(-c/b));
			}
			else {
				System.out.println("Ecuatie invalida!");
			}
			
		}
		
		/// delta == Math.sqrt(2); 
	}

}

package curs2;

import java . util .*;


public class Radiators {
	
	static final Scanner input = new Scanner ( System . in ) ;

	public static void main(String[] args) {
		
		double lungime, latime, inaltime;
		
		System.out.print("Lungimea camerei=");
		lungime = input . nextDouble () ;
		
		System.out.print("Latimea camerei=");
		latime  = input . nextDouble () ;
		
		System.out.print("Inaltimea camerei=");
		inaltime = input . nextDouble () ;
		
		System.out.println(lungime);
		System.out.println(latime);
		System.out.println(inaltime);

		double Vol = lungime*latime*inaltime;
		System.out.println("Volumul       = "+Vol);
		
		
		System.out.println("Nr radiatoare = "+Vol/8.0);

	}
}

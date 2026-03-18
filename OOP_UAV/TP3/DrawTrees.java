package test;

import java.util.*;


public class DrawTrees {
	
	static final Scanner input = new Scanner(System.in);


	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		int h;
		
		System.out.print("Introduceti un nr intreg: ");
		h = input.nextInt();
		
		//System.out.println(h);
		
		for(int i=1; i<= h; i++) {
			//System.out.print(i);
			
			// Spatiile!! 
			for(int j=1 ; j<=h-i; j++) {
				System.out.print(" ");
			}
			
			// Stelutele
			for(int j=1 ; j<=2*i-1; j++) {
				System.out.print("*");
			}
			
			System.out.print("\n");
		}
		
	}

}

package test;

public class BooleanTypes {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		int i = 1;
		int j = 2;
		int k;
		
		
		//System.out.println(i==j);
		//System.out.println(i = j);

		// Testarea diferentei
		//System.out.println(i != j);

		//System.out.println(true || false);

		// Ternary operator
		
		i = -1;
		
		k = ( i>=0 ? i : -i );
		
		if (i>=0) {
			k = i;
		} else {
			k =-i;
		}
		
		System.out.println(k);
		
		
		
	}

}

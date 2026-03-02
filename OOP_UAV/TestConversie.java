package curs1;

public class TestConversie {
	
	public static String miroir(String S ) {
		if (S.equals("")) {
			return "";
		}
		return miroir(S.substring(1)) + S.charAt(0);
	}
	
	public static void main(String[] args) {
		
		float x = 123456789;
		
		System.out.println(x); // pierdere de precizie
		
		float y = 1.234f;     // literal
		
		String s = "Hello";
		System.out.println(s);
		
		// Concatenarea se efectueaza cu +
		String s2 = s+" world!";
		System.out.println(s2+y);
		
		// Descoperim metodele disponibile adaugand .
		System.out.println(s.charAt(1));
		
		System.out.println(10 == 10/3 * 3 + 10%3);
		System.out.println("10/3 ca si int = "+10/3); // impartirea 2 intregi -> intreg
		
		System.out.println("10%3="+10%3);

		int n;
		System.out.println((n=3)==3);
		System.out.println(n);
		
		int i=3;
		System.out.println((i *= 2)==6);
		
		System.out.println(i);

		System.out.println(++i);
		System.out.println(i);

		final double pi = 3.14159;
		System.out.println(pi);
		
		x = -1.13f;
		if ( x >= 0) {
			y = x;
		} else {
			y = -x;
		}
		System.out.println("x="+x+"    y="+y);
		
		System.out.println("Switch");
		i = 2;
		switch (2*i) {
			case 4:
				System.out.println(" much ") ;
			case 1+1:
				System.out.println (" too much ") ;
			default:
				System.out.println("altceva");
		}
		
		// Definim un array de float
		float f[];
				
		f = new float[4];
		f[0] = 10.10f;
		f[1] = 20.3f;
		f[2] = 230.3f;
		f[3] = 10.3f;
		
		i = 0; // int
		while (i<4) {
			System.out.println(i+"   "+f[i]);
			i = i+1;
		}
		
		i = 0;
		n = 4; // lungimea lui f
		while ( i < n ) {
			System.out.print((( i == 0) ? "" : " ") + f[ i ]) ;
			i += 1;
		}
		System . out . println();
		
		
		for(i = 0; i < n ; i=i+2) {
			System . out . print ((( i == 0) ? "" : " ") + f [ i ]) ;
		}
		
		
		System . out . println () ;
		System . out . println () ;

		
		for (i = 0; ; i++) {
			if (i == 5) {
				System . out . print (" jump and ") ; continue ;
			} else if ( i == 10) {
				System . out . print (" we ' re done ! ") ; break ;
			}
			System . out . print ( i + " and ") ;
		}
		System . out . println () ;
		
		
		System.out.println(s);
		System.out.println(miroir(s));
		
	}

}

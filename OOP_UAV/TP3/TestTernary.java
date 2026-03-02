package sesiunea3;

public class TestTernary {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		int x , y , z ;
		// ...
		
		x = 4;
		y = 6;
		
		// z = max(x,y)
		z = (x > y ? x : y );
		
		System.out.println("Valoarea lui z este "+z);
		
		if ( x > y )
			z = x;
		else
			z = y;
		System.out.println("Valoarea lui z este "+z);
		
	}

}

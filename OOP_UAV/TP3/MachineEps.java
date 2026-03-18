package test;

public class MachineEps {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		float eps = 1f;
		
		while ((1 + eps)!=1) {
			eps = eps/2;
		}; 
		
		System.out.println(eps);
		
	}

}

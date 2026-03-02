package sesiunea3;

public class Box {
	// ATTRIBUTES
	private boolean closed;
	private String name;
	private int value;
	//private String something;
	
	// CONSTRUCTORS
	public Box (String n , int v ) {
		this.name = n ; // this = obiectul actual, instanta curenta a clasei
		this.value = v ;
		this.closed = true;
	}
	// overloading pentru constructor! 
	public Box (int v ) {
		this("Standard",v) ; // this e un constructor
	}
	
	// Request
	public int continut() {
		return this.value;
	}
	public String name() {
		return this.name;
	}
	public boolean closed() {
		return this.closed;
	}
	
	// Methods
	public void open() {
		this.closed = false;
	}
	public void close() {
		this.closed = true;
	}
	
	// Modificarea valorii 
	public void fill(int v) {
				if (this.closed()) {
					throw new AssertionError();
				}
				this.value = v;
	}
	
}

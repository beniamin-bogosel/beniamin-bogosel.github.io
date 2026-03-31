package test;

public class Box {
	// ATRIBUTE
	private boolean closed;
	private String name;
	private int value;
	private String pass = "Parola Tare";
	
	//CONSTRUCTOR
	public Box(String n, int v) {
		this.name = n;
		this.value = v;
		this.closed = true;
	}
	
	// REQUEST
	public boolean closed() {
		return this.closed;
	}
	public int contents(){
	    return this.value;
	}
	public String getPass() {
		return pass;
	}
	
	// METHODS
	public void open() {
		this.closed = false;
	}
	public void close() {
		this.closed = true;
	}
	
	
	public void clear(String passWord) {
		if (pass.equals(passWord)) {
			if (closed()) {
				open();
			}
			value = 0;
			close();
		} else {
			System.out.println("Ha Ha!");
		}
	}
}

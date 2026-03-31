package test;

public class Box {
	
	// ATRIBUTE
	private boolean closed;
	private String name;
	private int value;
	
	// Constructori
	public Box(String n, int v) {
		this.name = n;
		this.value = v;
		this.closed = true;
	}
	public Box(int v) {
		//this.name = "Standard";
		//this.value = v;
		//this.closed = true;
		// Sau apelam constructorul precedent
		this("Standard",v);
	}
	
	// Requests
	public String name() {
		return this.name;
	}
	public int contents() {
		return this.value;
	}
	public boolean State() {
		return this.closed;
	}
	
	
	// Method
	public void ChangeName(String NewName) {
		// Conditii
		this.name = NewName;
	}
	public void open() {
		this.closed = false;
	}
	public void close() {
		this.closed = true;
	}
	public void fill(int v) {
		if(this.closed) {
			throw new AssertionError();
		}
		this.value = v;
	}
}

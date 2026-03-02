package test;

public class Student {
	// Atrributes
	private int id;
	private String name;
	private int grade;
	// private in gradeAltCurs; 
	
	// Constructor
	public Student(String name,  int id, int grade) {
		this.name  = name;
		this.grade = grade;
		this.id    = id;
	}
	// Constructor overloaded: folosim doar numele
	public Student(String name,int id) {
		this(name,id,0);
	}
	public Student(String name) {
		this(name,0,0);
	}
	public Student() {
		this("no_name",0,0);
	}
	
	// accesarea datelor
	public int getId() {
		return this.id;
	}
	public String getName() {
		return this.name;
	}
	public int getGrade() {
		return this.grade;
	}
	
	// modificarea datelor 
	// 
	public void setId(int id) {
		this.id = id;
	}
	public void setGrade(int grade) {
		this.grade = grade;
	}
	public void setName(String name) {
		// if (name nu verifica cerintele pentru un "NUME"
		// atunci throw an Exception
		// si vom zice utilizatorului sa introduca un nume valid
		this.name = name;
	}
	
}

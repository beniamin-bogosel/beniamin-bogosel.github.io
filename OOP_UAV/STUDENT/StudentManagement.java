package test;

import java.util.*;

public class StudentManagement {
	
	static final Scanner input = new Scanner (System.in);
	
	private static ArrayList<Student> allstudents = new ArrayList<Student>();
	
	public static void addStudent() {
		System.out.println("Introduceti numele studentului");
		String name, surname;
		name = input.next();
		System.out.println("Introduceti prenumele studentului");
		surname = input.next();
		System.out.println("Numele a fost capturat: "+name+" "+ surname);
		
		int CNP;
		System.out.println("Introduceti CNP-ul (ID) studentului");
		CNP = input.nextInt();
		
		// ar trebui sa verificam daca CNP-ul nu e deja in lista
		
		Student s = new Student(name,CNP);
		allstudents.add(s);
	}
	
	public static void listStudents() {
		// Loop peste toate elementele unui ArrayList
		int i=1;
		for(Student stud : allstudents){ 
			System.out.println(i+" "+stud.getName()); 
			i++;
		}
	}
	
	public static void listGrades() {
		// Loop peste toate elementele
		for(Student s : allstudents){ 
			System.out.println(s.getName());
			System.out.println("   Nota = "+s.getGrade());
		}
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		//addStudent();
	
	    Student s1 = new Student("Popescu",0);
	    allstudents.add(s1);
	    
	    Student s2 = new Student("Ionescu",0,12345);
	    allstudents.add(s2);
	    
	    Student s3 = new Student("Popa",0,12345);
	    
	    allstudents.add(s3);
	    
	    Student s4 = new Student();
	    allstudents.add(s4);
	    
	    System.out.println(allstudents.size());
	    
	    
	    
	    //addStudent();
	    
	    //listStudents();
	    //listGrades();
	    
	    // Interfata cu utilizatorul
	    
	    boolean cont = true;
	    while (cont) {
	    		System.out.println("=================================================");
	    		System.out.println("Optiuni disponibile");
	    		System.out.println("1. Adaugare Student");
	    		System.out.println("2. Listarea tuturor Numelor Studentilor");
	    		System.out.println("3. Listarea notelor");
	    		System.out.println("4. Iesire din program");

	    		System.out.println("Introduceti optiunea dorita (ca si nr. intreg)");
	    		
	    		int option=0;
	    		//try {
	    			option = input.nextInt();
	    		//} catch (InputMismatchException e) {
	    			// zone treating the other cases
	    		//	System.out.println("Nu ati introdus o optiune valida! ");
	    		//	System.out.println("Introduceti un nr intreg! ");
	    		//}
	    		
	    		
	    		switch (option) {
	    			case 1: {
	    	    			System.out.println("Start Procedura Adaugare Student");
	    	    			addStudent();
	    	    			break;
	    			}
	    			case 2: {
    	    				System.out.println("Start Procedura Listare Studenti");
    	    				listStudents();
    	    				break;
	    			}
	    			case 3: {
	    				System.out.println("Listare Note Studenti");
	    				listGrades();
	    				break;
    			    }
	    			case 4: {
	    				System.out.println("Se executa iesirea din program");
	    				cont = false;
	    				break;
	    			}
	    			default: {
	    				System.out.println("Introduceti un nr integ ce corespunde unei optiuni.");
	    			}
	    			
	    		}
	    		
	    		if (cont==false) {
	    			break;
	    		}
	    		System.out.println("=================================================");

	    }
	    
	}

}

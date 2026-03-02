package test;

import javax.swing.*;
import java.awt.* ;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class SumaDouaNumere extends JFrame implements ActionListener {
	
	private static JTextField text1=null;
	private static JTextField text2=null;

	private static JLabel label=null;
	private static JButton button=null;
	private static int number = 0;
	
	static final String Alfabet = "abcdefghijklmnopqrstuvwxyz";
	
	public SumaDouaNumere() {
		// Instead of JFrame frame= new JFrame("test");
		super("Count Clicks"); // calls constructor from JFrame
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.setSize(600,600);
		this.setLayout(new GridLayout(2,1));
		JPanel panel1 = new JPanel();
		
		
		text1 = new JTextField();
		text2 = new JTextField();

		JLabel label1 = new JLabel("+");
		
		panel1.setLayout(new GridLayout(1,3));
		panel1.add(text1);
		panel1.add(label1);
		panel1.add(text2);
		//panel.setSize(300,300);
		//panel.add(label);
		this.add(panel1);
		
		JPanel panel2 = new JPanel();
		button = new JButton("Buton");
		button.addActionListener(this);

		
		panel2.setLayout(new GridLayout(2,1));

		label = new JLabel("Rezultat");
		panel2.add(button);
		panel2.add(label);
		
		this.add(panel2);

		

	}
	
	public void actionPerformed(ActionEvent e){
		if(e.getSource()==button){
			String t1=text1.getText();
			String t2=text2.getText();
			
			// convert string to int
			int n1 = Integer.parseInt(t1);
			int n2 = Integer.parseInt(t2);
			
			System.out.println("Primul = "+t1);
			System.out.println("Al Doilea = "+t2);
			number++;
			label.setText("Rezultat = "+(n1+n2));
		}
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		JFrame frame=new SumaDouaNumere();
		frame.setVisible(true);

	}

}

package test;

import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

//Trebuie adaugat: requires java.desktop;
//in module-info.java din Proiectul vostru
import javax.swing.*;

public class Adaugare extends JFrame implements ActionListener{
	
	private static JTextField text=null;
	private static JLabel label=null;
	private static JButton button=null;
	private static int number = 0;
	
	public Adaugare() {
		// Instead of JFrame frame= new JFrame("test");
		super("Adaugare"); // calls constructor from JFrame
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.setSize(600,600);
		JPanel panel = new JPanel();
		//JLabel label=new JLabel("Bonjour tout le monde");
		//panel.add(label);
		//panel.setLayout(new FlowLayout(FlowLayout.CENTER));
		//this.setLayout(new FlowLayout());
		
		//JPanel pan = new JPanel();
		label = new JLabel("");
		button = new JButton("Buton");
		text = new JTextField();
		button.addActionListener(this);
		panel.setLayout(new GridLayout(3,1));
		//panel.setLayout(new FlowLayout());

		panel.add(text); // on ajoute le JTextField text au JPanel pan
		panel.add(button);
		panel.add(label);
		this.add(panel);

	}
	
	public void actionPerformed(ActionEvent e){
		if(e.getSource()==button){
			//String a=text.getText();
			number++;
			String s = text.getText();
			System.out.println(s); // afisarea lui s in terminal
			label.setText("Textul introdus este: "+s);

		}
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		JFrame frame = new Adaugare();
		frame.setVisible(true);
	}

}

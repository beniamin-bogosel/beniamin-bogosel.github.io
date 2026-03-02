package test;

import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

//Trebuie adaugat: requires java.desktop;
//in module-info.java din Proiectul vostru
import javax.swing.*;
import java.awt.* ;

public class GrilaButoane extends JFrame implements ActionListener {
	
	private static JTextField text=null;
	private static JLabel label=null;
	private static JButton button=null;
	private static int number = 0;
	
	public GrilaButoane() {
		// Instead of JFrame frame= new JFrame("test");
		super("Count Clicks"); // calls constructor from JFrame
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.setSize(600,600);
		JPanel panel = new JPanel();
		//JLabel label=new JLabel("Bonjour tout le monde");
		//panel.add(label);
		//panel.setLayout(new FlowLayout(FlowLayout.CENTER));
		//this.setLayout(new FlowLayout());
		
		//JPanel pan = new JPanel();
		//label = new JLabel("");
		button = new JButton("Buton");
		JButton button1 = new JButton("Buton1");
		JButton button2 = new JButton("Buton2");
		JButton button3 = new JButton("Buton3");
		JButton button4 = new JButton("Buton4");
		JButton button5 = new JButton("Buton5");
		JButton button6 = new JButton("Buton6");
		JButton button7 = new JButton("Buton7");
		JButton button8 = new JButton("Buton8");
		JButton button9 = new JButton("Buton9");
		JButton button10 = new JButton("Buton10");
		JButton button11 = new JButton("Buton11");
		JButton button12 = new JButton("Buton12");

		// Array ce contine 12 butoane
		JButton[] TabelButoane = new JButton[24];
		
		button6.setText("Text Modificat");
		
		button.addActionListener(this);
		panel.setLayout(new GridLayout(4,3));
		//panel.setLayout(new FlowLayout());
		panel.setLayout(new FlowLayout(FlowLayout.CENTER));

		//System.out.println(TabelButoane.length);
		
		for(int i=0; i < TabelButoane.length; i++) {
			TabelButoane[i] = new JButton("Buton"+(i+1));
			panel.add(TabelButoane[i]);
		}

		//panel.setSize(300,300);
		//panel.add(label);
		this.add(panel);

	}
	
	public void actionPerformed(ActionEvent e){
		if(e.getSource()==button){
			//String a=text.getText();
			number++;
			label.setText("Number of clicks = "+number);
		}
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		JFrame frame=new GrilaButoane();
		frame.setVisible(true);

	}

}

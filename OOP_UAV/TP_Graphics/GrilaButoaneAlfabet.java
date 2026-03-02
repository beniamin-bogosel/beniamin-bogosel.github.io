package test;

//Trebuie adaugat: requires java.desktop;
//in module-info.java din Proiectul vostru
import javax.swing.*;
import java.awt.* ;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class GrilaButoaneAlfabet extends JFrame implements ActionListener {
	
	private static JTextField text=null;
	private static JLabel label=null;
	private static JButton button=null;
	private static int number = 0;
	
	static final String Alfabet = "abcdefghijklmnopqrstuvwxyz";
	
	public GrilaButoaneAlfabet() {
		// Instead of JFrame frame= new JFrame("test");
		super("Count Clicks"); // calls constructor from JFrame
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.setSize(600,600);
		JPanel panel = new JPanel();
		
		button = new JButton("Buton");
		

		// Array ce contine x butoane
		JButton[] TabelButoane = new JButton[Alfabet.length()];
				
		button.addActionListener(this);
		panel.setLayout(new GridLayout(4,3));
		//panel.setLayout(new FlowLayout());
		panel.setLayout(new FlowLayout(FlowLayout.CENTER));

		//System.out.println(TabelButoane.length);
		
		for(int i=0; i < TabelButoane.length; i++) {
			//String si = ""; 
			TabelButoane[i] = new JButton(""+Alfabet.charAt(i));
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
				
		JFrame frame=new GrilaButoaneAlfabet();
		frame.setVisible(true);

	}

}

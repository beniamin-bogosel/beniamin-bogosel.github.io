package test;

// Trebuie adaugat: requires java.desktop;
// in module-info.java din Proiectul vostru
import javax.swing.*;
import java.awt.event.*;
import java.awt.*;

public class CountClicks extends JFrame implements ActionListener {
	
	private static JTextField text=null;
	private static JLabel label=null;
	private static JButton button=null;
	private static int number = 0;
	
	public CountClicks() {
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
			label.setText("Number of clicks = "+number);
		}
	}
	
	public static void main(String[] args) {
		JFrame frame=new CountClicks();
		frame.setVisible(true);
		
		
	}
		
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package mazeducks;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;
import javax.swing.table.DefaultTableModel;

/**
 *
 * @author Medha Joshi
 */
public class Timeboard extends javax.swing.JFrame {

    /**
     * Creates new form Arcadeboard
     */
    public Timeboard() {
        initComponents();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        leaderboard_pane = new javax.swing.JScrollPane();
        leaderboard_table = new javax.swing.JTable();
        jButton1 = new javax.swing.JButton();
        time = new javax.swing.JButton();
        jLabel2 = new javax.swing.JLabel();
        jLabel1 = new javax.swing.JLabel();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setMaximumSize(new java.awt.Dimension(800, 600));
        setMinimumSize(new java.awt.Dimension(800, 600));
        setPreferredSize(new java.awt.Dimension(800, 600));
        setSize(new java.awt.Dimension(800, 600));
        getContentPane().setLayout(null);

        leaderboard_table.setModel(new javax.swing.table.DefaultTableModel(
            new Object [][] {

            },
            new String [] {
                "Name", "Score", "Time"
            }
        ));
        leaderboard_pane.setViewportView(leaderboard_table);

        getContentPane().add(leaderboard_pane);
        leaderboard_pane.setBounds(200, 200, 440, 280);

        jButton1.setText("Back");
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });
        getContentPane().add(jButton1);
        jButton1.setBounds(20, 50, 59, 32);

        time.setBackground(new java.awt.Color(255, 255, 255));
        time.setFont(new java.awt.Font("Champagne & Limousines", 0, 24)); // NOI18N
        time.setText("Show");
        time.setBorder(null);
        time.setBorderPainted(false);
        time.setContentAreaFilled(false);
        time.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                timeActionPerformed(evt);
            }
        });
        getContentPane().add(time);
        time.setBounds(390, 130, 47, 28);

        jLabel2.setFont(new java.awt.Font("Champagne & Limousines", 1, 48)); // NOI18N
        jLabel2.setText("Leaderboard");
        getContentPane().add(jLabel2);
        jLabel2.setBounds(270, 40, 280, 60);

        jLabel1.setIcon(new javax.swing.ImageIcon(getClass().getResource("/mazeducks/leaderboard.png"))); // NOI18N
        getContentPane().add(jLabel1);
        jLabel1.setBounds(0, 0, 800, 620);

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void timeActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_timeActionPerformed
        // TODO add your handling code here:
        
        try
    {
        Class.forName("com.mysql.cj.jdbc.Driver");
        Connection con = DriverManager.getConnection("jdbc:mysql://127.0.0.1:3306/mazegame?zeroDateTimeBehavior=convertToNull", "root", "1234");
        String query1 = "select * from timeboard order by score/((hrs*3600)+(mins*60)+secs) desc;";
        Statement st=con.createStatement();
        ResultSet rs=st.executeQuery(query1);
        DefaultTableModel model=(DefaultTableModel)leaderboard_table.getModel(); 
        model.setRowCount(0);
        Object [] row = new Object [3];
        while(rs.next())
        {
            
            row[0] = rs.getString("name");
            row[1] = rs.getString("score");
            row[2] = rs.getString("hrs") + ":" + rs.getString("mins") + ":" + rs.getString("secs");
            model.addRow(row);
        }
    }
    catch (Exception e)
    {
        System.out.print(e);
    }
    }//GEN-LAST:event_timeActionPerformed

    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
        Timed T = new Timed();
        this.dispose();
        T.setVisible(true);
    }//GEN-LAST:event_jButton1ActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(Arcadeboard.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(Arcadeboard.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(Arcadeboard.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(Arcadeboard.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new Arcadeboard().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton jButton1;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JScrollPane leaderboard_pane;
    private javax.swing.JTable leaderboard_table;
    private javax.swing.JButton time;
    // End of variables declaration//GEN-END:variables
}

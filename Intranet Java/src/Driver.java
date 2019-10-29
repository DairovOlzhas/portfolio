import javax.xml.crypto.Data;
import java.lang.invoke.SwitchPoint;
import java.util.Scanner;
import java.util.Vector;

public class Driver {


    public static Scanner reader = new Scanner(System.in);
    public static Database d = new Database();
    public static void main(String [] args){
        Database.deserialize();
     //   System.out.println(Database.getStudents().elementAt(0));
//        Vector<Admin> admins = new Vector<>();
//        admins.addElement(new Admin("IDSI", Gender.FEMALE, "Dairov", "Olzhas", "87472555792", "dairov.official#gmail.com", 152));
//        Database.setAdmins(admins);
      //  System.out.println(Database.getCourses().elementAt(0).getStudents());
        Start();
        Database.serialize();
    }

    private static void Start() {
        System.out.println("Welcome to University System!!!");
        preLogin();
    }

    private static void preLogin() {
        System.out.println("1) Authorization");
        System.out.println("2) Exit");
        System.out.print("#__");

        String in = reader.nextLine();
        if(in.equals("1")) {


            System.out.print("LOGIN:__");
            String log = reader.nextLine();
            System.out.print("Password:__");
            String pass = reader.nextLine();

            for (int i = 0; i < Database.getTeachers().size(); i++)
                if (pass.equals(Database.getTeachers().elementAt(i).getPassword()) && log.equals(Database.getTeachers().elementAt(i).getUsername())) {
                    TeacherDriver.menu(Database.getTeachers().elementAt(i));
                    preLogin();
                    return;
                }

            for (int i = 0; i < Database.getAdmins().size(); i++)
                if (pass.equals(Database.getAdmins().elementAt(i).getPassword()) && log.equals(Database.getAdmins().elementAt(i).getUsername())) {
                    AdminDriver.menu(Database.getAdmins().elementAt(i));
                    preLogin();
                    return;
                }
            for (int i = 0; i < Database.getManagers().size(); i++)
                if (pass.equals(Database.getManagers().elementAt(i).getPassword()) && log.equals(Database.getManagers().elementAt(i).getUsername())) {
                    ManagerDriver.menu(Database.getManagers().elementAt(i));
                    preLogin();
                    return;
                }
            for (int i = 0; i < Database.getStudents().size(); i++)
                if (pass.equals(Database.getStudents().elementAt(i).getPassword()) && log.equals(Database.getStudents().elementAt(i).getUsername())) {
                    StudentDriver.menu(Database.getStudents().elementAt(i));
                    preLogin();
                    return;
                }
            for (int i = 0; i < Database.getExecutors().size(); i++)
                if (pass.equals(Database.getExecutors().elementAt(i).getPassword()) && log.equals(Database.getExecutors().elementAt(i).getUsername())) {
                    ExecutorDriver.menu(Database.getExecutors().elementAt(i));
                    preLogin();
                    return;
                }
            System.out.println("Bad password or login!! Try again!!");
            preLogin();
        }
    }
}

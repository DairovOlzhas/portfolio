import javax.print.attribute.standard.OrientationRequested;
import javax.xml.crypto.Data;
import javax.xml.datatype.DatatypeConfigurationException;
import java.awt.image.AreaAveragingScaleFilter;
import java.io.*;
import java.util.ArrayList;
import java.util.Vector;

public class Database implements Serializable {
    private static Vector<Student> students;
    private static Vector<Teacher> teachers;
    private static Vector<Admin> admins;
    private static Vector<Executor> executors;
    private static Vector<Manager> managers;
    private static Vector<Order> orders;
    private static Vector<Order> donedOrders;
    private static Vector<Course> courses;


    public Database(){
        students = new Vector<>();
        teachers = new Vector<>();
        executors = new Vector<>();
        admins = new Vector<>();
        managers = new Vector<>();
        orders = new Vector<>();
        donedOrders = new Vector<>();
        courses = new Vector<>();
    }


    public static void serialize() {
        teachersSerialize();
        studentsSerialize();
        managersSerialize();
        adminsSerialize();
        executorsSerialize();
        ordersSerialize();
        donedOrdersSerialize();
        coursesSerialize();
    }
    public static void deserialize() {
        teachersDeserialize();
        studentsDeserialize();
        managersDeserialize();
        adminsDeserialize();
        executorsDeserialize();
        ordersDeserialize();
        donedOrdersDeserialize();
        coursesDeserialize();
    }

    public static void teachersSerialize() {
        try {
            ObjectOutputStream ois = new ObjectOutputStream(new FileOutputStream("Teachers.out"));
            ois.writeObject(Database.teachers);
            ois.close();
        }catch (FileNotFoundException fn){

        }catch (IOException io){

        }

    }
    public static void teachersDeserialize() {
        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream("Teachers.out"));
            Database.teachers = (Vector<Teacher>) ois.readObject();
            ois.close();
        }catch (FileNotFoundException fn){

        }catch (IOException io){

        }catch (ClassNotFoundException cnfe){

        }
    }

    public static void studentsSerialize() {
        try {
            ObjectOutputStream ois = new ObjectOutputStream(new FileOutputStream("Students.out"));
            ois.writeObject(Database.students);
            ois.close();
        }catch (FileNotFoundException fn){
            System.out.println("File not found student");

        }catch (IOException io){
            System.out.println("IO student");
        }

    }
    public static void studentsDeserialize() {
        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream("Students.out"));
            Database.students = (Vector<Student>) ois.readObject();
            ois.close();
        }catch (FileNotFoundException fn){
            System.out.println("not fount student");
        }catch (IOException io){
            System.out.println("io student des");
        }catch (ClassNotFoundException cnfe){
            System.out.println("class not found student");
        }
    }

    public static void managersSerialize() {
        try {
            ObjectOutputStream ois = new ObjectOutputStream(new FileOutputStream("Managers.out"));
            ois.writeObject(Database.managers);
            ois.close();
        }catch (FileNotFoundException fn){

        }catch (IOException io){

        }

    }
    public static void managersDeserialize() {
        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream("Managers.out"));
            Database.managers = (Vector<Manager>) ois.readObject();
            ois.close();
        }catch (FileNotFoundException fn){

        }catch (IOException io){

        }catch (ClassNotFoundException cnfe){

        }
    }

    public static void adminsSerialize() {
        try {
            ObjectOutputStream ois = new ObjectOutputStream(new FileOutputStream("Admins.out"));
            ois.writeObject(Database.admins);
            ois.close();
        }catch (FileNotFoundException fn){

        }catch (IOException io){

        }

    }
    public static void adminsDeserialize() {
        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream("Admins.out"));
            Database.admins = (Vector<Admin>) ois.readObject();
            ois.close();
        }catch (FileNotFoundException fn){
            System.out.println("io student des");

        }catch (IOException io){
            System.out.println("io afmin des");

        }catch (ClassNotFoundException cnfe){

        }
    }

    public static void executorsSerialize() {
        try {
            ObjectOutputStream ois = new ObjectOutputStream(new FileOutputStream("Executors.out"));
            ois.writeObject(Database.executors);
            ois.close();
        }catch (FileNotFoundException fn){

        }catch (IOException io){

        }

    }
    public static void executorsDeserialize() {
        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream("Executors.out"));
            Database.executors = (Vector<Executor>) ois.readObject();
            ois.close();
        }catch (FileNotFoundException fn){

        }catch (IOException io){

        }catch (ClassNotFoundException cnfe){

        }
    }

    public static void ordersSerialize() {
        try {
            ObjectOutputStream ois = new ObjectOutputStream(new FileOutputStream("Orders.out"));
            ois.writeObject(Database.orders);
            ois.close();
        }catch (FileNotFoundException fn){

        }catch (IOException io){

        }

    }
    public static void ordersDeserialize() {
        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream("Orders.out"));
            Database.orders = (Vector<Order>) ois.readObject();
            ois.close();
        }catch (FileNotFoundException fn){

        }catch (IOException io){

        }catch (ClassNotFoundException cnfe){

        }
    }

    public static void donedOrdersSerialize() {
        try {
            ObjectOutputStream ois = new ObjectOutputStream(new FileOutputStream("DonedOrders.out"));
            ois.writeObject(Database.donedOrders);
            ois.close();
        }catch (FileNotFoundException fn){

        }catch (IOException io){

        }

    }
    public static void donedOrdersDeserialize() {
        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream("DonedOrders.out"));
            Database.donedOrders = (Vector<Order>) ois.readObject();
            ois.close();
        }catch (FileNotFoundException fn){

        }catch (IOException io){

        }catch (ClassNotFoundException cnfe){

        }
    }

    public static void coursesSerialize() {
        try {
            ObjectOutputStream ois = new ObjectOutputStream(new FileOutputStream("Courses.out"));
            ois.writeObject(Database.courses);
            ois.close();
        }catch (FileNotFoundException fn){

        }catch (IOException io){

        }

    }
    public static void coursesDeserialize() {
        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream("Courses.out"));
            Database.courses = (Vector<Course>) ois.readObject();
            ois.close();
        }catch (FileNotFoundException fn){

        }catch (IOException io){

        }catch (ClassNotFoundException cnfe){

        }
    }

    static Vector<Admin> getAdmins() { return admins; }
    public static void setAdmins(Vector<Admin> admins) { Database.admins = admins; Database.adminsSerialize();}

    static Vector<Executor> getExecutors() { return executors; }
    public static void setExecutors(Vector<Executor> executors) { Database.executors = executors; Database.executorsSerialize();}

    static Vector<Student> getStudents() { return students; }
    public static void setStudents(Vector<Student> students) { Database.students = students; Database.studentsSerialize();}

    static Vector<Teacher> getTeachers() { return teachers; }
    public static void setTeachers(Vector<Teacher> teachers) { Database.teachers = teachers; Database.teachersSerialize();}

    public static Vector<Manager> getManagers() { return managers; }
    public static void setManagers(Vector<Manager> managers) { Database.managers = managers; Database.managersSerialize();}

    static Vector<Order> getDonedOrders() { return donedOrders; }
    public static void setDonedOrders(Vector<Order> donedOrders) { Database.donedOrders = donedOrders; Database.donedOrdersSerialize();}

    static Vector<Order> getOrders() { return orders; }
    public static void setOrders(Vector<Order> orders) { Database.orders = orders; Database.ordersSerialize();}

    public static Vector<Course> getCourses() { return courses; }
    public static void setCourses(Vector<Course> courses) { Database.courses = courses; Database.coursesSerialize();}



    @Override
    public String toString() {
        return "Number of: \n" + this.getClass()  + ";"
                + "; orders - " + orders.size()
                + "; doned orders - " + donedOrders.size();
    }
}

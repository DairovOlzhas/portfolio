import java.io.Serializable;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Vector;

public abstract class Employee extends User implements Salariable, Serializable {
    private int salary;

    Employee(String id, Gender gender, String lastname, String firstname, String telNumber, String email, int salary){
        super(id, gender, lastname, firstname, telNumber, email);
        this.salary = salary;
    }

    public void setSalary(int salary) { this.salary = salary; }

    void sendOrder(Order order){
        Vector<Order> orders = Database.getOrders();
        orders.addElement(order);
        Database.setOrders(orders);
    }

    public int getSalary(){ return salary; }

}

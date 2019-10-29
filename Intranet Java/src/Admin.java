import javax.xml.crypto.Data;
import java.util.Vector;

public class Admin extends Employee {
    public Admin(String id, Gender gender, String lastname, String firstname, String telNumber, String email, int salary){
        super(id, gender, lastname, firstname, telNumber, email, salary);
    }

    public void remAdmin(Admin admin){
        Vector<Admin> admins = Database.getAdmins();
        admins.removeElement(admin);
        Database.setAdmins(admins);
    }
    public void addAdmin(Admin admin) {
        Vector<Admin> admins = Database.getAdmins();
        admins.addElement(admin);
        Database.setAdmins(admins);
    }

    public void remExecutor(Executor executor){
        Vector<Executor> executors = Database.getExecutors();
        executors.removeElement(executor);
        Database.setExecutors(executors);
    }
    public void addExecutor(Executor executor) {
        Vector<Executor> executors = Database.getExecutors();
        executors.addElement(executor);
        Database.setExecutors(executors);
    }

    public void remStudent(Student student){
        Vector<Student> students = Database.getStudents();
        students.removeElement(student);
        Database.setStudents(students);

        Vector<Course> courses = Database.getCourses();
        for(int i = 0; i < courses.size(); i++){
            if(courses.elementAt(i).getStudents().removeElement(student)){
                students = courses.elementAt(i).getStudents();
                students.removeElement(student);
                courses.elementAt(i).setStudents(students);
            }
        }
        Database.setCourses(courses);
    }
    public void addStudent(Student student){
        Vector<Student> students = Database.getStudents();
        students.addElement(student);
        Database.setStudents(students);
    }

    public void remTeacher(Teacher teacher){
        Vector<Teacher> teachers = Database.getTeachers();
        teachers.removeElement(teacher);
        Database.setTeachers(teachers);

        Vector<Course> courses = Database.getCourses();
        for(int i = 0; i < courses.size(); i++){
            if(courses.elementAt(i).getTeacher().equals(teacher)){
                courses.elementAt(i).setTeacher(null);
            }
        }
        Database.setCourses(courses);
    }
    public void addTeacher(Teacher teacher){
        Vector<Teacher> teachers = Database.getTeachers();
        teachers.addElement(teacher);
        Database.setTeachers(teachers);
    }

    public void remManager(Manager manager) {
        Vector<Manager> managers = Database.getManagers();
        managers.removeElement(manager);
        Database.setManagers(managers);
    }
    public void addManager(Manager manager) {
        Vector<Manager> managers = Database.getManagers();
        managers.addElement(manager);
        Database.setManagers(managers);
    }

    @Override
    public String toString() {
        return "Admin{ username = " + getUsername()
                + "}";
    }
}

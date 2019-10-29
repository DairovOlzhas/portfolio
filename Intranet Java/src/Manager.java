import javax.xml.crypto.Data;
import java.util.Vector;

public class Manager extends Employee {

    Manager(String id, Gender gender, String lastname, String firstname, String telNumber, String email, int salary){
        super(id, gender, lastname, firstname, telNumber, email, salary);
    }


    public static void addCourse(Course course) {
        Vector<Course> courses = Database.getCourses();
        courses.addElement(course);
        Database.setCourses(courses);

        Vector<Teacher> teachers = Database.getTeachers();
        for(int i = 0 ; i < teachers.size(); i++){
            if(teachers.elementAt(i).equals(course.getTeacher())){
                courses = teachers.elementAt(i).getCourses();
                courses.addElement(course);
                teachers.elementAt(i).setCourses(courses);
            }
        }
        Database.setTeachers(teachers);
        Database.coursesSerialize();
        Database.teachersSerialize();
    }

    public static void remCourse(Course course) {
        Vector<Course> courses = Database.getCourses();
        courses.removeElement(course);
        Database.setCourses(courses);
        Vector<Student> students = Database.getStudents();
        for(int i = 0 ; i < students.size(); i++){
            if(students.elementAt(i).getTakenCourses().removeElement(course)){
                courses = students.elementAt(i).getTakenCourses();
                courses.removeElement(course);
                students.elementAt(i).setTakenCourses(courses);
            }
        }
        Database.setStudents(students);

        Vector<Teacher> teachers = Database.getTeachers();
        for(int i = 0 ; i < teachers.size(); i++){
            if(teachers.elementAt(i).equals(course.getTeacher())){
                courses = teachers.elementAt(i).getCourses();
                courses.removeElement(course);
                teachers.elementAt(i).setCourses(courses);
            }
        }
        Database.setTeachers(teachers);
        Database.teachersSerialize();
        Database.coursesSerialize();
        Database.studentsSerialize();
    }

    @Override
    public String toString() {
        return "Manager{ username = " + getUsername()
                + "}";
    }
}

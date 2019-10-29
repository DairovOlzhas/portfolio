import javax.xml.crypto.Data;
import java.util.*;

public class Teacher extends Employee{

    private Vector<Course> courses;
    private Position position;

    public Teacher(String id, Gender gender, String lastname, String firstname, String telNumber, String email, int salary, Position position){
        super(id, gender, lastname, firstname, telNumber, email, salary);
        this.position = position;
        courses = new Vector<>();
    }

    public Vector<Course> getCourses() { return courses; }
    public void setCourses(Vector<Course> courses) { this.courses = courses; }

    public Position getPosition() { return position; }
    public void setPosition(Position position) { this.position = position; }

    public void addCourseFile(Course course, CourseFile courseFile) {
        Vector<CourseFile> courseFiles = course.getCourseFiles();
        courseFiles.addElement(courseFile);
        course.setCourseFiles(courseFiles);
    }
    public void remCourseFile(Course course, CourseFile courseFile) {
        Vector<CourseFile> courseFiles = course.getCourseFiles();
        courseFiles.removeElement(courseFile);
        course.setCourseFiles(courseFiles);
    }

    public void addMark(Course course, Mark mark, Student student) {
        int index = Database.getStudents().indexOf(student);
        System.out.println(Database.getStudents());
        System.out.println(student);
        System.out.println("index of student ar " + index);
        HashMap<Course, Vector<Mark>> marks = student.getMarks();
        Vector<Mark> m = (student.getMarks().get(course) == null) ? new Vector<>():student.getMarks().get(course);
        m.addElement(mark);
        marks.put(course, m);
        student.setMarks(marks);
        Vector<Student> students = Database.getStudents();
        students.setElementAt(student, index);
        Database.setStudents(students);
    }
    public void setMark(Course course, Mark fromMark, Mark toMark, Student student) {
        int index = Database.getStudents().indexOf(student);
        HashMap<Course, Vector<Mark>> marks = student.getMarks();
        marks.get(course).setElementAt(toMark, marks.get(course).indexOf(fromMark));
        student.setMarks(marks);
        Vector<Student> students = Database.getStudents();
        students.setElementAt(student, index);
        Database.setStudents(students);
    }
    public void remMark(Course course, Mark mark, Student student) {
        int index = Database.getStudents().indexOf(student);
        HashMap<Course, Vector<Mark>> marks = student.getMarks();
        marks.get(course).removeElement(mark);
        student.setMarks(marks);
        Vector<Student> students = Database.getStudents();
        students.setElementAt(student, index);
        Database.setStudents(students);
    }


    @Override
    public boolean equals(Object obj) {
        if(this == obj)
            return true;
        if(obj == null)
            return false;
        if(obj.getClass() != getClass())
            return false;
        Teacher other = (Teacher) obj;
        if (courses == null) {
            if (other.courses != null)
                return false;
        } else if (!courses.equals(other.courses))
            return false;
        if (position == null) {
            if (other.position!= null)
                return false;
        } else if (!position.equals(other.position))
            return false;
        return true;
    }

    @Override
    public String toString() {
        return "Teacher{" +
                "courses=" + courses +
                ", position=" + position +
                '}';
    }
}

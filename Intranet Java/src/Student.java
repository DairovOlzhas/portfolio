import javax.xml.crypto.Data;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

public class Student extends User implements Serializable{

    private Degree degree;
    private int yearOfStudy;
    private String faculty;
    private String specialization;
    private Transcript transcript;
    private Vector<Course> takenCourses;
    private HashMap<Course, Vector<Mark>> marks;

    public Student(String id, Gender gender, String lastname, String firstname, String telNumber, String email, String faculty, String specialization, int yearOfStudy, Degree degree){
        super(id, gender, lastname, firstname, telNumber, email);
        this.degree = degree;
        this.yearOfStudy = yearOfStudy;
        this.faculty = faculty;
        this.specialization = specialization;
        transcript = new Transcript(this);
        takenCourses = new Vector<>();
        marks = new HashMap<>();
    }

    public Transcript getTranscript() {
        transcript.refreshTranscript(this);
        return transcript;
    }

    public Degree getDegree() { return degree; }
    public void setDegree(Degree degree) { this.degree = degree; }

    public String getFaculty() { return faculty; }
    public void setFaculty(String faculty) { this.faculty = faculty; }

    public int getYearOfStudy() { return yearOfStudy; }
    public void setYearOfStudy(int yearOfStudy) { this.yearOfStudy = yearOfStudy; }

    public String getSpecialization() { return specialization; }
    public void setSpecialization(String specialization) { this.specialization = specialization; }

    public Vector<Course> getTakenCourses() { return takenCourses; }
    public void setTakenCourses(Vector<Course> takenCourses) { this.takenCourses = takenCourses; }

    public void TakeCourse(Course course) {
        Course chCourse = course;

        Vector<Student> students = course.getStudents();
        students.addElement(this);
        chCourse.setStudents(students);

        takenCourses.addElement(chCourse);

        Vector<Course> courses = Database.getCourses();
        courses.setElementAt(chCourse, courses.indexOf(course));
        Database.setCourses(courses);

        Teacher teacher = course.getTeacher();
        courses = teacher.getCourses();

        courses.setElementAt(chCourse, courses.indexOf(course));
        teacher.setCourses(courses);



    }


    public HashMap<Course, Vector<Mark>> getMarks() { return marks; }
    public void setMarks(HashMap<Course, Vector<Mark>> marks) { this.marks = marks; }


    @Override
    public boolean equals(Object obj) {
        if(this == obj)
            return true;
        if(obj == null)
            return false;
        if(obj.getClass() != getClass())
            return false;
        Student other = (Student) obj;
        if (degree == null) {
            if (other.degree != null)
                return false;
        } else if (!degree.equals(other.degree))
            return false;
        if (specialization == null) {
            if (other.specialization != null)
                return false;
        } else if (!specialization.equals(other.specialization))
            return false;
        if (transcript == null) {
            if (other.transcript != null)
                return false;
        } else if (!transcript.equals(other.transcript))
            return false;
        if (faculty == null) {
            if (other.faculty!= null)
                return false;
        } else if (!faculty.equals(other.faculty))
            return false;
        if (takenCourses == null) {
            if (other.takenCourses!= null)
                return false;
        } else if (!takenCourses.equals(other.takenCourses))
            return false;
        if (marks == null) {
            if (other.marks!= null)
                return false;
        } else if (!marks.equals(other.marks))
            return false;
        return yearOfStudy == other.yearOfStudy;
    }

    @Override
    public String toString() {
        return "Student{" +
                "degree=" + degree +
                ", yearOfStudy=" + yearOfStudy +
                ", faculty='" + faculty + '\'' +
                ", specialization='" + specialization + '\'' +
                ", transcript=" + transcript +
                ", takenCourses=" + takenCourses +
                ", marks=" + marks +
                '}';
    }
}

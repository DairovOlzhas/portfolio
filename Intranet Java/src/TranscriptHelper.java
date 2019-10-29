import java.io.Serializable;

public class TranscriptHelper implements Serializable {
    private String courseName;
    private double gpa;
    private Grade grade;
    private int points;

    TranscriptHelper(Student student, Course course){
        courseName = course.getCourseName();
        points = 0;
        for(int i = 0; i < student.getMarks().get(course).size(); i++){
            points += student.getMarks().get(course).elementAt(i).getPoint();
        }
        int cnt = (points - 45)/5;
        if(cnt != 0)
            gpa = 1 + 3/cnt;
        grade = Grade.values()[cnt];
    }

    public String getCourseName() { return courseName; }
    public double getGpa() { return gpa; }
    public Grade getGrade() { return grade; }
    public int getPoints() { return points; }

    @Override
    public boolean equals(Object obj) {
        if(this == obj)
            return true;
        if(obj == null)
            return false;
        if(obj.getClass() != getClass())
            return false;
        TranscriptHelper other = (TranscriptHelper) obj;
        if (gpa != other.gpa)
            return false;
        if (points != other.points)
            return false;
        if (courseName == null) {
            if (other.courseName!= null)
                return false;
        } else if (!courseName.equals(other.courseName))
            return false;
        if (grade == null) {
            if (other.grade!= null)
                return false;
        } else if (!grade.equals(other.grade))
            return false;
        return true;
    }

    @Override
    public String toString() {
        return courseName + " " + points + " " + gpa + " " + grade;
    }
}

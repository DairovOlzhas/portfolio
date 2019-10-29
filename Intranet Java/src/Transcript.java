import java.io.Serializable;
import java.util.Vector;

public class Transcript implements Serializable {

    private String studentName;
    private double totalGPA;
    private int totalCredits;
    private Vector<TranscriptHelper> transcriptHelpers;

    public Transcript(Student student){
        studentName = student.getFirstname() + " " + student.getLastname();
        totalGPA = 0;
        totalCredits = 0;
        transcriptHelpers = new Vector<>();
    }

    void refreshTranscript(Student student){
        for(int i = 0 ; i < student.getTakenCourses().size(); i++){
            transcriptHelpers.addElement(new TranscriptHelper(student, student.getTakenCourses().elementAt(i)));
            totalGPA += transcriptHelpers.elementAt(i).getGpa();
            totalCredits += student.getTakenCourses().elementAt(i).getCredit();
        }
        totalGPA /= totalCredits;
    }

    public double getTotalGPA() { return totalGPA; }
    public int getTotalCredits() { return totalCredits; }
    public String getStudentName() { return studentName; }
    public Vector<TranscriptHelper> getTranscriptHelpers() { return transcriptHelpers; }

    @Override
    public boolean equals(Object obj) {
        if(this == obj)
            return true;
        if(obj == null)
            return false;
        if(obj.getClass() != getClass())
            return false;
        Transcript other = (Transcript) obj;
        if (totalGPA != other.totalGPA)
            return false;
        if (totalCredits != other.totalCredits)
            return false;
        if (studentName == null) {
            if (other.studentName!= null)
                return false;
        } else if (!studentName.equals(other.studentName))
            return false;
        if (transcriptHelpers == null) {
            if (other.transcriptHelpers!= null)
                return false;
        } else if (!transcriptHelpers.equals(other.transcriptHelpers))
            return false;
        return true;
    }

    @Override
    public String toString() {
        String s = "";
        s += studentName + " Total GPA:" + totalGPA + " Credits: " + totalCredits + "\n";
        for(int i = 0; i < transcriptHelpers.size(); i++)
            s += transcriptHelpers.elementAt(i)+ "\n";
        return s;
    }
}


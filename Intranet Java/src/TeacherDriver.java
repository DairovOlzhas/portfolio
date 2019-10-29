import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.Date;
import java.util.HashMap;

public class TeacherDriver implements Serializable {
    static Teacher t;
    public static void menu(Teacher teacher){
        System.out.println("Welcome " + teacher.getFirstname()+" "+teacher.getLastname()+"!!!");
        t = teacher;
        Teach_menu();
    }
    private static void Teach_menu(){

        System.out.println("1. Information about me");
        System.out.println("2. Courses");
        System.out.println("3. Log out");
        System.out.print("#__");
        String ans = Driver.reader.nextLine();

        switch (ans.charAt(0)){
            case('1'):
                getInfo();
                break;
            case('2'):
                getCourse();

                break;
//            case('3'):
//                return;
        }
    }

    private static void getInfo(){
        System.out.println("Full information('b' - to back):" + t.toString());
        System.out.print("#__");
        String in = Driver.reader.nextLine();
        if(in.equals("b")){
            Teach_menu();
        }
    }
    private static void getCourse(){
        System.out.println("Courses List :");
        for(int i = 0 ; i < t.getCourses().size();i++){
            System.out.println((i+1)+". "+t.getCourses().get(i).getCourseName());
        }
        System.out.println("Please choose the Course or press 'q' to quite");
        System.out.print("#___");
        String in = Driver.reader.nextLine();
        if(in.equals("q") || t.getCourses().isEmpty()){
            Teach_menu();
        }
        else
            courseContent(in);

    }
    private static void courseContent(String st){
        System.out.println("1. List of Students");
        System.out.println("2. Course Files");
        System.out.println("3. Put Marks");
        System.out.println("4. Add file");
        System.out.println("5. Remove file");
        System.out.print("#__");
        String in = Driver.reader.nextLine();
        switch(in.charAt(0)){
            case('1'):
                getListStudent(st);
                break;
            case('2'):
                getCourseFiles(st);
                break;
            case('3'):
                putMarks(st);
                break;
            case('4'):
                addFile(st);
                break;
            case('5'):
                removeFile(st);
                break;

        }
        Teach_menu();
    }

    private static void getListStudent(String it){
        for(int i = 0 ; i < t.getCourses().get(Integer.parseInt(it)-1).getStudents().size();i++) {
            System.out.println((i + 1) + ". " + t.getCourses().get(Integer.parseInt(it) - 1).getStudents().get(i).getFirstname() + " " +
                    t.getCourses().get(Integer.parseInt(it) - 1).getStudents().get(i).getLastname() + "Point out of 100" +
                    t.getCourses().get(Integer.parseInt(it) - 1).getStudents().get(i).getMarks().get(t.getCourses().get(Integer.parseInt(it) - 1)));
        }
        System.out.println("Press 'q' to quite");
            System.out.print("#__");
            String ans = Driver.reader.nextLine();
            if(ans.equals("q")){
                courseContent(it);
            }

        }


    private static void putMarks(String it){
        for(int i = 0; i < t.getCourses().get(Integer.parseInt(it)-1).getStudents().size();i++) {
            System.out.println((i + 1) + ". " + t.getCourses().get(Integer.parseInt(it) - 1).getStudents().get(i).getFirstname() + " " +
                    t.getCourses().get(Integer.parseInt(it) - 1).getStudents().get(i).getLastname());
        }
            System.out.println("Choose the Student number to put Mark or Press 'q' to quite");
            System.out.print("#___");
            String ans = Driver.reader.nextLine();
            if(ans.equals("q")){
                courseContent(it);
            }
            int IndexOfStudent = Integer.parseInt(ans)-1;
            int IndexOfSubject = Integer.parseInt(it)-1;
            putMarkStudent(IndexOfSubject,IndexOfStudent);
    }

    private static void putMarkStudent(int indSubject,int indStudent){
        System.out.println("Enter Mark range of 0 - 100 ");
        System.out.print("#__");
        String ans = Driver.reader.nextLine();
        Mark m = new Mark(Integer.parseInt(ans),new Date());
        t.addMark(t.getCourses().elementAt(indSubject),m,t.getCourses().elementAt(indSubject).getStudents().elementAt(indStudent));
    }
    private static void addFile(String it){
        String f_name = Driver.reader.nextLine();
        System.out.println("Please write name of your file");
        System.out.print("#__ ");
        String file_name = Driver.reader.nextLine();
        System.out.println("Please write link(way) file");
        System.out.print("#__ ");
        String way = Driver.reader.nextLine();
        File f = new File(way);
        CourseFile c_f = new CourseFile(file_name,f);
        t.addCourseFile(t.getCourses().get(Integer.parseInt(it)-1), c_f);
    }
    private static void removeFile(String it){
        for(int i = 0 ; i < t.getCourses().get(Integer.parseInt(it)-1).getCourseFiles().size();i++){
            System.out.println((i+1)+". " + t.getCourses().get(Integer.parseInt(it)-1).getCourseFiles().get(i).getFileName());
        }
        System.out.println("Choose number course file");
        System.out.print("#__");
        String ans = Driver.reader.nextLine();
        t.remCourseFile(t.getCourses().get(Integer.parseInt(it)-1), t.getCourses().get(Integer.parseInt(it)-1).getCourseFiles().elementAt(Integer.parseInt(ans)-1));
    }
    private static void getCourseFiles(String it){
        for(int i = 0 ; i < t.getCourses().get(Integer.parseInt(it)-1).getCourseFiles().size();i++){
            System.out.println((i+1)+". " + t.getCourses().get(Integer.parseInt(it)-1).getCourseFiles().get(i).getFileName());
        }
    }
}
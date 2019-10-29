import javax.xml.crypto.Data;

public class StudentDriver {
    static Student s;
    public static void menu(Student student) {
        s = student;
        System.out.println("------STUDENT MENU------");
        System.out.println("Welcome " + student.getFirstname() + "!!");
        System.out.println();
        Actions();


    }

    private static void Actions() {
        System.out.println("Choose action( q - to quit):");
        System.out.println("1) Information about me");
        System.out.println("2) Registration to Course");
        System.out.println("3) My Courses");
        System.out.println("4) View Transcript");

        System.out.print("__");
        String com = Driver.reader.nextLine();

        switch (com){
            case "1":
                showInformation();
                break;
            case "2":
                regToCourse();
                break;
            case "3":
                viewCourses();
                break;
            case "4":
                viewTrans();
                break;
            case "q":
                return;
        }

        Actions();
    }

    private static void viewTrans() {
        System.out.println(s.getTranscript());
    }

    private static void viewCourses() {
        System.out.println("Choose course to view(b - to back):");

        for(int i = 0; i < Database.getCourses().size(); i++){
            System.out.println((i+1) + ") " + Database.getCourses().elementAt(i).getCourseName());
        }

        System.out.println("#___");
        String in = Driver.reader.nextLine();

        if(in.equals("b")){
            Actions();
            return;
        }

        viewCourse(Database.getCourses().elementAt(Integer.parseInt(in)-1));
    }

    private static void viewCourse(Course course) {
        System.out.println("Choose action(b - to back):");
        System.out.println("1) View Marks");
        System.out.println("2) CourseFiles");

        System.out.println("#___");
        String in = Driver.reader.nextLine();

        if(in.equals("b")){
            viewCourses();
            return;
        }

        switch (in){
            case"1":
                System.out.println("Marks : ");
                for(int i = 0; i < s.getMarks().get(course).size(); i++){
                    System.out.println(s.getMarks().get(course));
                }
                viewCourses();
                return;
            case"2":
                System.out.println("Course Files : ");
                for(int i = 0; i < course.getCourseFiles().size(); i++){
                    System.out.println((i+1) + ") " + course.getCourseFiles().elementAt(i).getFileName());
                }
                viewCourses();
                return;
        }

    }

    private static void regToCourse() {
        System.out.println("Choose course to Registration(b - to back):");

        for(int i = 0; i < Database.getCourses().size(); i++){
            System.out.println((i+1) + ") " + Database.getCourses().elementAt(i).getCourseName());
        }

        System.out.println("#___");
        String in = Driver.reader.nextLine();
        if(in.equals("b")){
            Actions();
            return;
        }

        s.TakeCourse(Database.getCourses().elementAt(Integer.parseInt(in)-1));
        System.out.println("Course taken successfully!!");

        Actions();
    }


    private static void showInformation() {
        System.out.println(s);

        System.out.print(" [1] to back__");
        String com = Driver.reader.nextLine();

        switch (com){
            case "1":
                Actions();
                break;
            default:
                showInformation();
                break;
        }
    }
}

import javax.xml.crypto.Data;

public class ManagerDriver {
    static Manager m;

    public static void menu(Manager manager) {
        System.out.println("Welcome : " +manager.getFirstname()+" "+manager.getLastname());
        m = manager;
        Manager_Menu();
    }

    private static void Manager_Menu(){
        System.out.println("Please print the number of action");
        System.out.println();
        System.out.println("1. Courses");
        System.out.println("2. Log out");
        System.out.print("#__");
        String ans = Driver.reader.nextLine();

        switch (ans.charAt(0)){
            case '1':
                coursePage();
                break;
//            case '2':
//                return;
        }
    }

    private static void coursePage(){
        System.out.println("Please print the number of action");
        System.out.println();
        System.out.println("1. Create course");
        System.out.println("2. Course list");
        System.out.println("3. Remove course");
        System.out.println("4. go Back");
        System.out.print("#__");
        String ans = Driver.reader.nextLine();

        switch (ans.charAt(0)){
            case '1':
                addingCourse();
                break;
            case '2':
                courseList();
                break;
            case '3':
                removingCourse();
                break;
            case '4':
                Manager_Menu();
                break;

        }
    }

    private static void removingCourse() {
        System.out.println("Choose course to removing( 'b' - to back):");
        for(int i = 0 ; i < Database.getCourses().size(); i++){
            System.out.println((i+1) + ") " + Database.getCourses().elementAt(i).getCourseName());
        }

        System.out.println("#___");
        String in = Driver.reader.nextLine();

        if(in.equals("b")){
            coursePage();
            return;
        }

        m.remCourse(Database.getCourses().elementAt(Integer.parseInt(in)-1));

        System.out.println("Course removed successfully!!");
        coursePage();

    }

    private static void addingCourse(){
        System.out.println("Please fill the Course fields with spaces as in Example");
        System.out.print("Course name:__");

        String name = Driver.reader.nextLine();

        for(int i=0; i< Database.getTeachers().size();i++) {
            System.out.println(i+1+". "+Database.getTeachers().get(i).getLastname()+ " " + Database.getTeachers().get(i).getFirstname());
        }
        System.out.print("Choose teacher:__");
        String tech = Driver.reader.nextLine();
        System.out.print("Course credits:__");
        String credit = Driver.reader.nextLine();
        System.out.print("Course code:__");
        String code = Driver.reader.nextLine();

        System.out.print("Course specialization:__");
        String spec = Driver.reader.nextLine();

        Course c = new Course(name, code, Integer.parseInt(credit), Database.getTeachers().get(Integer.parseInt(tech)-1) );
        m.addCourse(c);
        System.out.println("Course added successfully!!");
        coursePage();
        //addElement(c);

    }
    private static void courseList()  {
        System.out.println("Course List( 'b' - to back):");
        for(int i=0;i< Database.getCourses().size();i++) {
            System.out.println(i+1+". CourseTitle: "+Database.getCourses().get(i).getCourseName()+"; Teacher: "+Database.getCourses().get(i).getTeacher().getLastname());
        }
        String txt = Driver.reader.nextLine();
        if(txt.equals("b")) {
            coursePage();
        }
    }
}

import javax.xml.crypto.Data;

public class ExecutorDriver {
    static Executor e;
    public static void menu(Executor executor) {
        e = executor;
        
        
        System.out.println("------STUDENT MENU------");
        System.out.println("Welcome " + executor.getFirstname() + "!!");
        System.out.println();
        Actions();
    }

    private static void Actions() {
        System.out.println("Choose action( q - to quit):");
        System.out.println("1) Information about me");
        System.out.println("2) Do order");
        System.out.println("3) View Orders");

        System.out.print("__");
        String com = Driver.reader.nextLine();

        switch (com){
            case "1":
                showInformation();
                break;
            case "2":
                doOrder();
                break;
            case "3":
                viewOrders();
                break;
            case "q":
                return;
        }

        Actions();
    }

    private static void viewOrders() {
        System.out.println("Orders: ");
        for (int i = 0; i < Database.getOrders().size(); i++)
            System.out.println((i+1) + ") " + Database.getOrders().elementAt(i));
        for (int i = 0; i < Database.getDonedOrders().size(); i++)
            System.out.println((i+1+Database.getOrders().size()) + ") " + Database.getDonedOrders().elementAt(i));
        String in = Driver.reader.nextLine();
        Actions();
    }

    private static void doOrder() {
        System.out.println("Choose order to do(b - to back):");

        for(int i = 0; i < Database.getOrders().size(); i++){
            System.out.println((i+1) + ") " + Database.getOrders().elementAt(i));
        }

        System.out.println("#___");
        String in = Driver.reader.nextLine();
        Database.ordersSerialize();
        Database.donedOrdersSerialize();
        if(in.equals("b")){
            Actions();
            return;
        }

        e.doOrder(Database.getOrders().elementAt(Integer.parseInt(in)-1));

        System.out.println("Order status doned!!");

        Actions();
    }

    private static void showInformation() {
        System.out.println(e);

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

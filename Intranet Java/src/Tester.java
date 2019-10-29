import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Vector;

public class Tester {
    static class DB{
        static Vector<Person> persons;
        public DB(){
            persons = new Vector<>();
        }

        public static Vector<Person> getPersons() { return persons; }

        public static void setPersons(Vector<Person> persons) { DB.persons = persons; }

        @Override
        public String toString() {
            String string = "";
            for(int i = 0 ; i < persons.size(); i++)
                string += persons.get(i) + "\n";

            return string;
        }
    }
    static class Person{
        String name;
        String id;

        public Person(String name, String id){
            this.id = id;
            this.name = name;
        }

        public String getId() { return id; }

        public String getName() { return name; }

        public void setId(String id) { this.id = id; }

        public void setName(String name) { this.name = name; }

        @Override
        public boolean equals(Object obj) {
            if(obj instanceof Person){
                Person p = (Person)obj;
                System.out.println("equals of Person for:" + p.name + " " + p.id);
                return id.equals(p.id) && name.equals(p.name);
            }
            return false;
        }

        @Override
        public String toString() {
            return name + " " + id + super.toString();
        }
    }

    public static void main(String[] args){
        DB db = new DB();
        Vector<Person> a1 = new Vector<>();
        AdminDriver.logger(new Teacher("asdf", Gender.MALE, "asdf", "asdf", "asdf", "asdf",25, Position.ASSISTENT), "added");
        DB.persons.addElement(new Person("DBperson", "id0"));
        DB.persons.addElement(new Person("DBperson", "id1"));
        DB.persons.addElement(new Person("DBperson", "id2"));
        DB.persons.addElement(new Person("DBperson", "id3"));
        DB.persons.removeElement(new Person("DBperson", "id4"));
        Vector<Person> a2 = new Vector<>(DB.getPersons());
        a1.addElement(DB.persons.elementAt(0));
        DB.persons.elementAt(0).setName("seted");

      //  DB.persons.insertElementAt(null, 0);
        DB.persons.setElementAt(null, 0);
        Person p = DB.getPersons().elementAt(1);
        p.setName("setted");
        System.out.println("asdf ::"+p);
        System.out.println("asdfasdf   ::" + DB.getPersons().elementAt(1));
       // DB.persons.

        System.out.print(a1 + " \n A2: \n" + a2);
        System.out.print("\n" + DB.persons + "\n");
    }
}

import java.io.Serializable;
import java.util.Date;

public abstract class User implements Serializable {
    private String id;
    private Gender gender;
    private String username;
    private String firstname;
    private String lastname;
    private String password;

    private String telNumber;

    private String email;
    public User(){}

    public User(String id, Gender gender, String lastname, String firstname, String telNumber, String email){
        this.id = id;
        this.gender = gender;
        username = firstname.charAt(0) + "_" + lastname;
        this.lastname = lastname;
        this.firstname = firstname;
        password = "kbtu111";
        this.telNumber = telNumber;
        this.email = email;
    }

    public String getId() { return id; }

    public Gender getGender() { return gender; }

    public String getUsername() { return username; }

    public String getLastname() { return lastname; }

    public void setLastname(String lastname) { this.lastname = lastname; }
    public String getFirstname() { return firstname; }

    public void setFirstname(String firstname) { this.firstname = firstname; }
    public String getPassword() { return password; }

    public void setPassword(String password) { this.password = password; }
    public String getTelNumber() { return telNumber; }

    public void setTelNumber(String telNumber) { this.telNumber = telNumber; }
    public String getEmail() { return email; }

    public void setEmail(String email) { this.email = email; }
    @Override
    public boolean equals(Object obj) {
        if(this == obj)
            return true;
        if(obj == null)
            return false;
        if(obj.getClass() != getClass())
            return false;
        User other = (User) obj;
        if (id == null) {
            if (other.id != null)
                return false;
        } else if (!id.equals(other.id))
            return false;
        if (gender == null) {
            if (other.gender != null)
                return false;
        } else if (!gender.equals(other.gender))
            return false;
        if (username == null) {
            if (other.username != null)
                return false;
        } else if (!username.equals(other.username))
            return false;
        if (firstname == null) {
            if (other.firstname != null)
                return false;
        } else if (!firstname.equals(other.firstname))
            return false;
        if (lastname == null) {
            if (other.lastname != null)
                return false;
        } else if (!lastname.equals(other.lastname))
            return false;
        if (password == null) {
            if (other.password != null)
                return false;
        } else if (!password.equals(other.password))
            return false;
        if (telNumber == null) {
            if (other.telNumber!= null)
                return false;
        } else if (!telNumber.equals(other.telNumber))
            return false;
        if (email == null) {
            if (other.email != null)
                return false;
        } else if (!email.equals(other.email))
            return false;
        return true;
    }


    @Override
    public String toString() {
        return "User{" +
                "id='" + id + '\'' +
                ", gender=" + gender +
                ", username='" + username + '\'' +
                ", firstname='" + firstname + '\'' +
                ", lastname='" + lastname + '\'' +
                ", password='" + password + '\'' +
                ", telNumber='" + telNumber + '\'' +
                ", email='" + email + '\'' +
                '}';
    }
}

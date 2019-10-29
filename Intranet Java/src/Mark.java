import java.io.Serializable;
import java.util.Date;
import java.util.Map;

public class Mark implements Serializable {
    private double point;
    private Date date;

    public Mark(double point, Date date){
        try {
            this.point = point;
            this.date = date;
            if(point > 30){
                throw new NotCorrectMarkException("Incorrect Mark");
            }
        }catch (NotCorrectMarkException e){

        }
    }

    public Date getDate() { return date; }
    public void setDate(Date date) { this.date = date; }

    public double getPoint() { return point; }
    public void setPoint(double point) { this.point = point; }

    @Override
    public boolean equals(Object obj) {
        if(this == obj)
            return true;
        if(obj == null)
            return false;
        if(obj.getClass() != getClass())
            return false;
        Mark other = (Mark) obj;
        if (date == null) {
            if (other.date != null)
                return false;
        } else if (!date.equals(other.date))
            return false;
        return point == other.point;
    }

    @Override
    public String toString() {
        return point + " " + date;
    }
}

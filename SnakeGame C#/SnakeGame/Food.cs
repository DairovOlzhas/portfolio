using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SnakeGame
{
    public class Food
    {
        public Point location;
        public char sign = '@';
        public ConsoleColor color = ConsoleColor.Green;

        public Food()
        {
            SetRandomFood();
        }

        public void Eat()
        {
            Point head = Game.snake.body[0];
            if (head.x == location.x && head.y == location.y)
            {
                SetRandomFood();
                Game.snake.body.Add(new Point(0, 0));
            }
        }

        public void SetRandomFood()
        {
            int x = new Random().Next(0, Game.sizex - 1);
            int y = new Random().Next(2, Game.sizey - 1);

            location = new Point(x, y);

            if(Game.Collision(location))
                SetRandomFood();              
        }
        
        public void Draw()
        {
            Console.ForegroundColor = color;            
            Console.SetCursorPosition(location.x, location.y);
            Console.Write(sign);
        }
    }
}

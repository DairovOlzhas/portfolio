import { Component, OnInit } from '@angular/core';
import { ApiService } from '../shared/services/api.service';
import { TaskList, Task } from '../shared/models/models';

@Component({
  selector: 'app-main',
  templateUrl: './main.component.html',
  styleUrls: ['./main.component.css']
})


export class MainComponent implements OnInit {

  public tasklists: TaskList[] = [];
  public tasklist: TaskList = {
      id: -1,
      name: ''
  };
  public tasks: Task[] = [];
  public task: Task = {
      id: -1,
      name: null,
      created_at: null,
      due_on: null,
      status: null,
      task_list_id: null
  };
  public now: Date;
  public newtask: Task = {
    id: null,
    name: '',
    created_at: this.now,
    due_on: null,
    status: 'not done',
    task_list_id: null
  };
  public activeTasklist = -1;
  public name: any = '';
  public logged = false;
  public login: any = '';
  public password: any = '';
  public searchquery = '';
  public maxdate: Date;
  public mindate: Date;
  public orderby = 'name';

  constructor(private provider: ApiService) { }

  ngOnInit() {
    const token = localStorage.getItem('token');
    if (token ) {
      this.logged = true;
    }

    if (this.logged) {
      this.getTaskLists();
    }
  }

  getTaskLists() {
    this.provider.getTaskLists().then( res => {
      this.tasks = [];
      this.tasklists = res;
    });
  }
  createTaskList() {
    if (this.name !== '') {
        this.provider.createTaskList(this.name).then(res => {
          console.log(res.name + ' tasklist created');
          this.name = '';
          this.tasklists.push(res);
        });
    }
  }

  getTaskList(tasklist: TaskList) {
    this.provider.getTaskList(tasklist).then(res => {
      if (tasklist !== this.tasklist) {
        this.task = {
            id: -1,
            name: null,
            created_at: null,
            due_on: null,
            status: null,
            task_list_id: null
        };
      }
      this.tasklist = res;
      console.log(this.task);
    });
  }
  updateTaskList(tasklist: TaskList) {
    console.log(tasklist.name + ' updated');
    this.provider.updateTaskList(tasklist).then(res => {
      this.tasklist = res;
      this.getTaskLists();
    });
  }
  deleteTaskList(tasklist: TaskList) {
    console.log(tasklist);
    this.provider.deleteTaskList(tasklist).then(res => {
      console.log(tasklist.name + ' deleted');
      this.tasklist = {
          id: -1,
          name: ''
      };
      this.getTaskLists();
    });
  }
  getTaskListsTasks(tasklist: TaskList) {
    this.provider.getTaskListsTask(tasklist).then(res => {
      this.tasks = res;
      this.getTaskList(tasklist);
    });
  }
  getTasks() {
    this.provider.getTasks().then(res => {
      this.tasks = res;
    });
  }
  createTask() {
    this.now = new Date();
    this.provider.createTask(this.newtask.name, this.now, this.newtask.due_on, this.newtask.status, this.tasklist.id).then(res => {
      this.getTaskListsTasks(this.tasklist);
      this.newtask = {
        id: null,
        name: '',
        created_at: null,
        due_on: null,
        status: 'not done',
        task_list_id: null
      };
    });
  }
  getTask(task: Task) {
    this.provider.getTask(task).then(res => {
      this.task = res;
    });
  }
  updateTask(task: Task) {
    this.provider.updateTask(task).then(res => {
      this.getTaskListsTasks(this.tasklist);
    });
  }
  deleteTask(task: Task) {
    this.provider.deleteTask(task).then(res => {
      this.getTaskListsTasks(this.tasklist);
      task = {
          id: -1,
          name: null,
          created_at: null,
          due_on: null,
          status: null,
          task_list_id: null
      };
    });
  }


  auth() {
    if (this.login !== '' && this.password !== '') {
      this.provider.auth(this.login, this.password).then(res => {
        localStorage.setItem('token', res.token);
        this.logged = true;
        this.getTaskLists();
      });
    }
  }

  logout() {
    this.provider.logout().then( res => {
      localStorage.clear();
      this.logged = false;
      this.tasks = [];
      this.tasklist = {
          id: -1,
          name: ''
      };
      this.task = {
          id: -1,
          name: null,
          created_at: null,
          due_on: null,
          status: null,
          task_list_id: null
      };
      this.tasklists = [];
    });
  }

  search() {
    this.provider.search(this.searchquery).then(res => {
      this.tasks = res;
    });
  }

  order() {
    this.provider.OrderBy(this.orderby).then(res => {
      this.tasks = res;
    });
  }

  filtrate() {
      console.log('filtreate' + this.mindate +'   ' +this.maxdate);
      let s = '';
      if (this.maxdate != null && this.mindate != null) {
          s = 'max_created_at=' + this.maxdate + '&min_created_at=' + this.mindate;
      } else if (this.mindate != null) {
          s = 'min_created_at=' + this.mindate;
      } else if (this.mindate != null) {
          s = 'max_created_at=' + this.maxdate;
      }
      this.provider.FiltrCreatedDate(s).then(res => {
        this.tasks = res;
      });
  }


}

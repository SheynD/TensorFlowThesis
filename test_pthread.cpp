#include <iostream>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

bool ready(false);
//A mutex for which to use locks 
std::mutex m;
//A condition variable
std::condition_variable cv;
//A counter to increment the state of the condvar
int i = 0;
//A mutex lock
pthread_mutex_t l1;

//Method for waiting for the lock to open up for the mutex
void waits() {
	std::unique_lock<std::mutex> lk(m);
	std::cout << "Waiting.. \n";
	cv.wait(lk, []{return i == 1;});
	std::cout << "...finished waiting. Incrementer was updated to 1\n";
	ready = true;
}

//A method for signaling when the lock is released and ready for use
void signals() {
	std::this_thread::sleep_for(std::chrono::seconds(2));
	std::cout << "Notifying falsely \n";
	cv.notify_one();
	std::cout << "after notify\n";	
	std::unique_lock<std::mutex> lk(m);
	i = 1;
	std::cout << "The status is now " << ready << std::endl;
	while(!ready){
		std::cout << "Notifying true change. \n";
		lk.unlock();
		cv.notify_one();
		std::this_thread::sleep_for(std::chrono::seconds(1));
		lk.lock();
	}
}

//Try performing locking and unlocking
void try_test() {
	std::cout << "locking..\n";
    pthread_mutex_lock(&l1);
	std::cout << "unlocking.. \n";
    pthread_mutex_unlock(&l1);
	std::cout << "Operation performed\n";
}

//Main method to run the program
int main() {
    std::cout << "Performing a simple lock test\n";
    pthread_mutex_init(&l1,NULL);
    pthread_mutex_lock(&l1);
    pthread_mutex_unlock(&l1);
    std::cout << "Done with simple lock test\n";
	try_test();
	try_test();
	try_test();    
//std::cout << "Doing with simple condvar test\n";
	//std::thread t1(waits), t2(signals);
	//t1.join();
	//t2.join();
}
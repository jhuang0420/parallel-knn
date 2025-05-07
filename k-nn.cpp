#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <random>
#include <chrono>
#include <queue>
#include <cstdint>
#include <thread>
#include <future>
#include <utility>
#include <algorithm>
#include <mutex>

using namespace std;

const size_t MAX_LEAFNODE_SIZE = 16;	// max size for points in a leafnode
const size_t SAMPLE_PERCENT = 5;	// sampling percent of total points for median
const size_t MIN_SAMPLE = 100;		// min sample to find median
const size_t MIN_SIZE_PARALLEL = 2048;	// min number of points to run parallel


uint64_t rand_gen() {
	random_device rd;
	mt19937_64 gen(rd());
	uniform_int_distribution<uint64_t> distrib(0, numeric_limits<uint64_t>::max());
	return distrib(gen);
}


class ThreadPool {
	public:
		// Constructor
		ThreadPool(size_t num_threads) {
			if (num_threads < 1) num_threads = 1;
			threads.reserve(num_threads); 	// alloc mem
			for (size_t i = 0; i < num_threads; i++) {
				threads.emplace_back([this] {
					for(;;) {
						function<void()> job;
						{	// scope block for mutex
							unique_lock<mutex> lock(this->queue_mutex);
							this->condition.wait(lock, [this] {return this->terminate || !this->jobs.empty();}); 	// waits for terminate or jobs queue is nonempty
							if(this->terminate && this->jobs.empty()) return;	// if terminate and queue is empty then end pool 
							job = move(this->jobs.front());	// moves job from queue to be executed 
							this->jobs.pop();	// removes job from queue
						}
						++active_jobs;
						job();
						--active_jobs;
					}
				});
			}
	
		}	


		// Enqueue job
		template<class F>
		auto enqueue(F&& f) -> future<decltype(f())>{

			using r_type = decltype(f());
			auto job = make_shared<packaged_task<r_type()>>(forward<F>(f));
			future<r_type> res = job->get_future();
			{
				unique_lock<mutex> lock(queue_mutex);
				jobs.emplace([job]() {(*job)();});
			}
			condition.notify_one();
			return res;
		}


		// Destructor
		~ThreadPool() {
			{
				unique_lock<mutex> lock(queue_mutex);
				terminate = true;			// set terminate to true
			}
			condition.notify_all();				// notify all threads
			for(thread &thread : threads) thread.join();	// joins all threads
		}


		bool is_idle() {
			return active_jobs < threads.size()-1;
		}


	private:
		bool terminate = false;		// flag to terminate pool
		mutex queue_mutex;		// mutex for sync	
		condition_variable condition;	// notif for avaliable jobs
		vector<thread> threads;		// threads to exec
		queue<function<void()>> jobs;	// jobs to be done
		atomic<int> active_jobs = 0;
};

struct kdnode {
	int split_dim;
	float split_val;
	unique_ptr<kdnode> left;
	unique_ptr<kdnode> right;

	bool isleaf;
	vector<vector<float>> points;

	kdnode(int dim, float val) : isleaf(false), split_dim(dim), split_val(val) {} 	// split node
	kdnode(vector<vector<float>> points) : isleaf(true), points(move(points)) {}	// leaf node
	
	// debug print
	void print(int depth = 0) const {
		string indent(depth*2, ' ');
		if (isleaf) {
			cout << indent << "Leaf (" << points.size() << " points):\n";
			for(const auto& pt: points) {
				cout << indent << "  [";
				for (float p : pt) cout << p << ", ";
				cout << "]\n";
			}
		} else {
			cout << indent << "Node (split_dim=" << split_dim << ", split_val=" << split_val << "):\n";
			if (left) left->print(depth+1);
			if (right) right->print(depth+1);
		}
	}
};


class kdtree {
	public: 
		kdtree(int num_cores): pool(num_cores*2) {} // initialize threadpool
		
		void build(vector<vector<float>> &points, int dimensions) {
			dim = dimensions;
			root = build_tree(move(points), 0);
		}	

		void knn(const vector<vector<float>> &q_points, int num_neighbors, ofstream &results) {
			vector<future<void>> tasks;
			vector<vector<vector<float>>> result(q_points.size());

			for(size_t i = 0; i < q_points.size(); i++) {
				tasks.push_back(pool.enqueue([this, i, &q_points, num_neighbors, &result] {
					auto neighbors = knn_search(q_points[i], num_neighbors);
					result[i] = move(neighbors);
				}));
			}
					
			for(auto &task : tasks) task.get();

			// write to results			
			for (auto &neighbors : result) {
				for (auto neighbor: neighbors) {
					for (float val : neighbor) {
						results.write(reinterpret_cast<char*>(&val), sizeof(val));
					}
				}
			}

		}

		// debug print
		void print() const {
			root->print();
		}
		
	private:
		unique_ptr<kdnode> root;
		int dim;
		ThreadPool pool;

		// calculate median
		float split(vector<vector<float>> &points, int split_dim) {
        		int size = points[0].size();
        		if (split_dim < 0 || split_dim >= size) exit(1);
        		vector<float> sample_points;
			
			size_t sample_size = max(MIN_SAMPLE, static_cast<size_t>(points.size()*SAMPLE_PERCENT/100));
        		
			size_t step = max(static_cast<size_t>(1), points.size() / sample_size); 
        		for (size_t i = 0; i < points.size(); i+=step) sample_points.push_back(points[i][split_dim]);
        		nth_element(sample_points.begin(), sample_points.begin() + sample_points.size()/2, sample_points.end());
        		return sample_points[sample_points.size()/2];
		}

		// build tree
		unique_ptr<kdnode> build_tree(vector<vector<float>> &&points, int curr_dim) {
			if (points.empty()) return nullptr;
			if (points.size() <= MAX_LEAFNODE_SIZE) return make_unique<kdnode>(move(points));	// make leafnode
			float x = split(points, curr_dim);						// make split node
			auto node =  make_unique<kdnode>(curr_dim, x);
			vector<vector<float>> left_points, right_points;
			for (vector<float> &p: points) {
				if (p[curr_dim] <= x) left_points.push_back(p);
				else right_points.push_back(p);
			}	
			
			// subtree building in parallel w/ pool
			future<unique_ptr<kdnode>> left_build, right_build;
			if (pool.is_idle() && left_points.size() > MIN_SIZE_PARALLEL) {
				left_build = pool.enqueue([this, left_points = move(left_points), curr_dim]() mutable{
					return build_tree(move(left_points),(curr_dim+1)%dim);});
				node->left = left_build.get();
			} else node->left = build_tree(move(left_points),(curr_dim+1)%dim);

			if (pool.is_idle() && right_points.size() > MIN_SIZE_PARALLEL) {
				right_build = pool.enqueue([this, right_points = move(right_points), curr_dim]() mutable{
					return build_tree(move(right_points),(curr_dim+1)%dim);});
				node->right = right_build.get();
			} else node->right = build_tree(move(right_points),(curr_dim+1)%dim);

			return node;
		}


		// knn search for each point
		vector<vector<float>> knn_search(const vector<float> &q_point, int num_neighbors) {
			priority_queue<pair<float, vector<float>>> neighbors;		// max heap
			knn_search_helper(root, q_point, num_neighbors, neighbors);
			
			vector<vector<float>> results(num_neighbors);
			int idx = num_neighbors-1;

			while(!neighbors.empty()) {	
				results[idx--] = move(neighbors.top().second);
				neighbors.pop();
			}
			
			return results;
		}

		// knn recursive
		void knn_search_helper(unique_ptr<kdnode> &node, const vector<float> &q_point, int num_neighbors, priority_queue<pair<float, vector<float>>> &neighbors) {
			if (!node) return;

			// add points to neighbor vector
			if (node->isleaf) {
				for(const vector<float> &p : node->points) {
					float dist = distance(q_point, p);
					if (neighbors.size() < num_neighbors) {
						neighbors.emplace(dist, p);
					} else if (dist < neighbors.top().first) {	// check if dist < max point in heap
						neighbors.pop();
						neighbors.emplace(dist, p);
					}
				}
			} else {
				bool left = q_point[node->split_dim] <= node->split_val;
				// traverse down tree
				if (left) {
					knn_search_helper(node->left, q_point, num_neighbors, neighbors);
				} else {
					knn_search_helper(node->right, q_point, num_neighbors, neighbors); 
				}

				// recurse other 
				if (neighbors.size() < num_neighbors || abs(q_point[node->split_dim] - node->split_val) < neighbors.top().first) {
					if (left) knn_search_helper(node->right, q_point, num_neighbors, neighbors);
					else knn_search_helper(node->left, q_point, num_neighbors, neighbors);	
				}
			}
		}

		// calculate euclidean distance
		float distance(const vector<float> &a, const vector<float> &b) {
			if (a.size() != b.size()) {
				cerr << "BAD CONTENT: different dimensions for distance calculation." << endl;
				return -1;	
			}
			float sum = 0.0;
			for (size_t i = 0; i < a.size(); i++) sum += pow(a[i]-b[i],2);
			return sqrt(sum);
		}

		

};


// ./k-nn n_cores training_file query_file result_file
int main(int argc, char **argv) {

	// Argument Validation
	if (argc != 5) {
		cerr << "Usage: ./k-nn n_cores training_file query_file result_file" << endl;
		return 1;
	}

	int cores = stoi(argv[1]);
	if (cores < 1) {
		cerr << "BAD_CONTENT: n_cores less than 1" << endl;
		return 1;
	}

	string training = argv[2], query = argv[3], result = argv[4];
	ifstream tf(training), qf(query), rf(result);

	size_t startID = 8;
	vector<char> tf_buffer(24);
	vector<char> qf_buffer(32);

	if (!tf.good()) {
		cerr << "NOT_FOUND: training_file" << endl;
		return 1;
	}
	// Read training file data
	if (tf.is_open()) {
		tf.seekg(startID);
		tf.read(tf_buffer.data(), 24);
	}

	if (!qf.good()) {
		cerr << "NOT_FOUND: query_file" << endl;
		return 1;
	}
	// Read query file data
	if (qf.is_open()) {
		qf.seekg(startID);
		qf.read(qf_buffer.data(), 32);
	}
	
	if (rf.good()) {
		cerr << "BAD_CONTENT: result_file already exists" << endl;
		return 1;
	}

	// Create results file
	ofstream results(result, ios::binary);
	if (!results.is_open()) {
		cerr << "Failed to open results file" << endl;
		return 1;
	}


	// Write the results header
	char fts[8] = "RESULT";
	results.write(fts, 8);			// File type string
	results.write(tf_buffer.data(), 8);	// Training file ID
	results.write(qf_buffer.data(), 8);	// Query file ID
	uint64_t rfID = rand_gen();
	results.write(reinterpret_cast<char*>(&rfID), 8); // Result file rand ID

	// Extract training data
	uint64_t t_points, t_dimensions;
	memcpy(&t_points, tf_buffer.data()+8, 8);
	memcpy(&t_dimensions, tf_buffer.data()+16, 8);
	
	vector<vector<float>> training_points(t_points, vector<float>(t_dimensions));	// Vector of all training points
	for(size_t i = 0; i < t_points; i++) {
		tf.read(reinterpret_cast<char*>(training_points[i].data()), t_dimensions * sizeof(float));
	}
	tf.close();

	// Extract query data
	uint64_t queries, q_dimensions, q_neighbors;
	memcpy(&queries, qf_buffer.data()+8, 8);
	memcpy(&q_dimensions, qf_buffer.data()+16, 8);
	memcpy(&q_neighbors, qf_buffer.data()+24, 8);

	vector<vector<float>> query_points(queries, vector<float>(q_dimensions));	// Vector of all query points
	for(size_t i = 0; i < queries; i++) {
		qf.read(reinterpret_cast<char*>(query_points[i].data()), q_dimensions * sizeof(float));
	}
	qf.close();

	if (t_dimensions != q_dimensions) {
		cerr << "Dimension mismatch" << endl;
		return 1;
	}

	results.write(reinterpret_cast<char*>(&queries),8); 	 // Number of queries
	results.write(reinterpret_cast<char*>(&q_dimensions),8); // Number of dimensions
	results.write(reinterpret_cast<char*>(&q_neighbors),8);  // Number of neighbors 		

	chrono::high_resolution_clock::time_point start, end;
	chrono::milliseconds duration;

	start = chrono::high_resolution_clock::now();

	// Build k-d tree with training_data vector w/ threadpool 
	kdtree tree(cores);
	tree.build(training_points, t_dimensions);
	// tree.print();

	end = chrono::high_resolution_clock::now();
	duration = chrono::duration_cast<chrono::milliseconds>(end-start);
	cout << "Time taken to build tree: " << duration.count() << " milliseconds" << endl;

	start = chrono::high_resolution_clock::now();
	// k-NN query computation & writing
	tree.knn(query_points, q_neighbors, results);

	results.close();
	
	end = chrono::high_resolution_clock::now();
	duration = chrono::duration_cast<chrono::milliseconds>(end-start);
	cout << "Time taken to query: " << duration.count() << " milliseconds" << endl;

	return 0;
}

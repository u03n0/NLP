#include <iostream>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <set>
#include <ranges>



int count_occurrences(const std::string& str, const std::string& sub) {
    int count {0};
    size_t pos {0};

    while ((pos = str.find(sub, pos)) != std::string::npos) {
        count++;
        pos += sub.length();
    }
    return count;
}

double tf(std::string d, std::string t) {
    std::stringstream ss(d);
    std::string word;
    std::vector<std::string> word_vector;

    while (ss >> word) {
        word_vector.push_back(word);
    }

    int n_terms = word_vector.size();
    int freq = {count_occurrences(d, t)};
    return (freq)? static_cast<double>(freq) / n_terms : 0.0;

}


double idf(std::vector<std::string> D, std::string t) {
    int count {0};

    for (const auto& d: D) {
        if (d.find(t)) {
            count++;
        }
    }
    return count;
}


std::string toLower(const std::string& str) {
    std::string lower_str = str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), [](unsigned char c){
        return std::tolower(c);
    });
    return lower_str;
}

std::set<std::string> buildVocabulary(const std::vector<std::string>& corpus) {
    std::set<std::string> vocab;  

    for (const std::string& doc : corpus) {
        std::istringstream stream(doc);  
        std::string word;
        
        while (stream >> word) {
            vocab.insert(toLower(word));
        }
    }

    return vocab;
}

std::vector<std::vector<double>> tf_idf(const std::vector<std::string>& D) {
    std::set<std::string> V = buildVocabulary(D);

    std::unordered_map<std::string, double> idfs;
    std::unordered_map<int, std::unordered_map<std::string, double>> tfs;

    for (const std::string& term : V) {
        idfs[term] = idf(D, term);
    }

    for (size_t idx = 0; idx < D.size(); ++idx) {
        for (const std::string& term : V) {
            tfs[idx][term] = tf(D[idx], term);
        }
    }

    std::vector<std::vector<double>> tfidf_matrix(D.size(), std::vector<double>(V.size(), 0.0));

    size_t col_idx = 0;
    for (const std::string& term : V) {
        for (size_t row_idx = 0; row_idx < D.size(); ++row_idx) {
            double tf_value = tfs[row_idx][term];
            double idf_value = idfs[term];
            tfidf_matrix[row_idx][col_idx] = tf_value * idf_value;
        }
        col_idx++;
    }

    return tfidf_matrix;
}

int main() {

    std::string term {"cat"};
    std:: string s1 {"The black cat and orange cat played together"};
    std:: string s2 {"The cat played with the red ball"};
    std::vector<std::string> D = {s1, s2};

   
    double tf_ {tf(s1, term)};
    double idf_ {idf(D, term)};

    std::vector<std::vector<double>> tfidf_matrix = tf_idf(D);

    std::cout << tf_ << std::endl;
    std::cout << idf_ << std::endl;

    for (const auto& row : tfidf_matrix) {
        for (double value : row) {
            std::cout << value << "\t";
        }
        std::cout << std::endl;
    }


    return 0;
}

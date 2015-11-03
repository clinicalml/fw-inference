#include "tb2wgrammar.hpp"   
#include <set>
#include <algorithm>

using namespace std;

void WCNFCFG::addVariableMeasure(int violationCost) {
            
    // Assume A_i -> a_i   
    // 1) for all prod. rules C->A_iB, add C->A_jB with weight = w[C->A_jB] + vioCost, j != i
    // 2) for all prod. rules C->BA_i, add C->BA_j with weight = w[C->A_jB] + vioCost, j != i
    set<int> nonTerms;
    for (vector<WCNFRule>::iterator p = termProd.begin();p != termProd.end();++p) {         
        nonTerms.insert(p->from);
    }                
    for (int i=0;i<2;i++) {
        vector<WCNFRule> prods(nonTermProd);
        for (vector<WCNFRule>::iterator p = prods.begin();p != prods.end();++p) {                 
            if (nonTerms.find(p->to[i]) != nonTerms.end()) {                           
                for (set<int>::iterator A = nonTerms.begin();A != nonTerms.end();++A) {         
                    if (*A != p->to[i]) {
                        WCNFRule rule = *p;
                        rule.to[i] = *A;
                        rule.weight += violationCost;
                        nonTermProd.push_back(rule);
                    }
                }
            }
         }                        
    }
    
    sort(nonTermProd.begin(), nonTermProd.end());
    nonTermProd.erase(unique(nonTermProd.begin(), nonTermProd.end()), nonTermProd.end());
                
}

void WCNFCFG::print(ostream &ofs) {
     for (vector<WCNFRule>::iterator p = nonTermProd.begin(); p != nonTermProd.end(); ++p) {
         ofs << p->from << "->" << p->to[0] << " " << p->to[1] << ": " << p->weight << "\n";
     }   
      for (vector<WCNFRule>::iterator p = termProd.begin(); p != termProd.end(); ++p) {
         ofs << p->from << "->" << p->to[0] << ": " << p->weight << "\n";
     }  
}
# 生成副本的模型变更

## decode/ln/A3h
```diff
```
## decode/ln/Gemini
```diff
```
## decode/mha/A3h
```diff
--- native formatted
+++ H/I generated
@@ -23,6 +23,7 @@
 from add_communication import add_beha_producers, build_event, add_mediumdata, add_broadcast
 from data_notation import tensor_notation
 from conv1d import Conv1d
+from hi_config import shared_history_conv1d
 from transpose import Transpose
 from rmsnorm import Rmsnorm
 from elerope import Elerope
@@ -116,7 +117,7 @@
 ]
 
 mha_split_parallelism_degree = (1, 8, 1, 1)
-heads = 7
+heads = 8
 
 q_parallelisms_list = [
     generate_average_degree(batch_size, mha_split_parallelism_degree[0]),
@@ -233,7 +234,7 @@
 
 p_next_oper_tag = v_next_oper_tag
 
-for head_idx in range(head_num):
+for head_idx in range(heads):
     q_next_oper_tag, next_online_data_tag, next_offline_data_tag, q_next_beha_offset, q_oper = Conv1d(
         data_dict=data_dict,
         beha_dict=beha_dict,
@@ -273,7 +274,7 @@
     )
     qrope_data_tag = qrope_oper.target_data_tags[0]
 
-    qkt_next_oper_tag, next_online_data_tag, next_offline_data_tag, qkt_next_beha_offset, qkt_oper = Conv1d(
+    qkt_next_oper_tag, next_online_data_tag, next_offline_data_tag, qkt_next_beha_offset, qkt_oper = shared_history_conv1d(
         data_dict=data_dict,
         beha_dict=beha_dict,
         oper_name="MHA"+"_qkt",
@@ -301,7 +302,7 @@
     )
     s_data_tag = s_oper.target_data_tags[0]        
 
-    p_next_oper_tag, next_online_data_tag, next_offline_data_tag, p_next_beha_offset, p_oper = Conv1d(
+    p_next_oper_tag, next_online_data_tag, next_offline_data_tag, p_next_beha_offset, p_oper = shared_history_conv1d(
         data_dict=data_dict,
         beha_dict=beha_dict,
         oper_name="MHA"+"_p",
```
## decode/mha/Gemini
```diff
--- native formatted
+++ H/I generated
@@ -23,6 +23,7 @@
 from add_communication import add_beha_producers, build_event, add_mediumdata, add_broadcast
 from data_notation import tensor_notation
 from conv1d import Conv1d
+from hi_config import shared_history_conv1d
 from transpose import Transpose
 from rmsnorm import Rmsnorm
 from elerope import Elerope
@@ -116,7 +117,7 @@
 ]
 
 mha_split_parallelism_degree = (1, 8, 1, 1)
-heads = 7
+heads = 8
 
 q_parallelisms_list = [
     generate_average_degree(batch_size, mha_split_parallelism_degree[0]),
@@ -233,7 +234,7 @@
 
 p_next_oper_tag = v_next_oper_tag
 
-for head_idx in range(head_num):
+for head_idx in range(heads):
     q_next_oper_tag, next_online_data_tag, next_offline_data_tag, q_next_beha_offset, q_oper = Conv1d(
         data_dict=data_dict,
         beha_dict=beha_dict,
@@ -273,7 +274,7 @@
     )
     qrope_data_tag = qrope_oper.target_data_tags[0]
 
-    qkt_next_oper_tag, next_online_data_tag, next_offline_data_tag, qkt_next_beha_offset, qkt_oper = Conv1d(
+    qkt_next_oper_tag, next_online_data_tag, next_offline_data_tag, qkt_next_beha_offset, qkt_oper = shared_history_conv1d(
         data_dict=data_dict,
         beha_dict=beha_dict,
         oper_name="MHA"+"_qkt",
@@ -301,7 +302,7 @@
     )
     s_data_tag = s_oper.target_data_tags[0]        
 
-    p_next_oper_tag, next_online_data_tag, next_offline_data_tag, p_next_beha_offset, p_oper = Conv1d(
+    p_next_oper_tag, next_online_data_tag, next_offline_data_tag, p_next_beha_offset, p_oper = shared_history_conv1d(
         data_dict=data_dict,
         beha_dict=beha_dict,
         oper_name="MHA"+"_p",
```
## decode/proj/A3h
```diff
```
## decode/proj/Gemini
```diff
```
## decode/ffn/A3h
```diff
```
## decode/ffn/Gemini
```diff
```
## prefill/ln/A3h
```diff
```
## prefill/ln/Gemini
```diff
```
## prefill/mha/A3h
```diff
--- native formatted
+++ H/I generated
@@ -116,7 +116,7 @@
 ]
 
 mha_split_parallelism_degree = (1, 4, 1, 1)
-heads = 7
+heads = 8
 
 q_parallelisms_list = [
     generate_average_degree(batch_size, mha_split_parallelism_degree[0]),
@@ -233,7 +233,7 @@
 
 p_next_oper_tag = v_next_oper_tag
 
-for head_idx in range(head_num):
+for head_idx in range(heads):
     q_next_oper_tag, next_online_data_tag, next_offline_data_tag, q_next_beha_offset, q_oper = Conv1d(
         data_dict=data_dict,
         beha_dict=beha_dict,
```
## prefill/mha/Gemini
```diff
--- native formatted
+++ H/I generated
@@ -115,7 +115,7 @@
 ]
 
 mha_split_parallelism_degree = (1, 4, 1, 1)
-heads = 7
+heads = 8
 
 q_parallelisms_list = [
     generate_average_degree(batch_size, mha_split_parallelism_degree[0]),
@@ -232,7 +232,7 @@
 
 p_next_oper_tag = v_next_oper_tag
 
-for head_idx in range(head_num):
+for head_idx in range(heads):
     q_next_oper_tag, next_online_data_tag, next_offline_data_tag, q_next_beha_offset, q_oper = Conv1d(
         data_dict=data_dict,
         beha_dict=beha_dict,
```
## prefill/proj/A3h
```diff
```
## prefill/proj/Gemini
```diff
```
## prefill/ffn/A3h
```diff
```
## prefill/ffn/Gemini
```diff
```

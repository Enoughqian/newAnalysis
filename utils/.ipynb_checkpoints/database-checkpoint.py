import os
import requests
import json
import faiss
import numpy as np
import mysql.connector
import traceback
import logging
from sklearn.decomposition import PCA
from config import DB_CONFIG, VECTOR_SEARCH_CONFIG

class VectorSearch:
    def __init__(self, dim, logger, target_dim=512):
        self.dim = dim
        self.target_dim = target_dim
        self.logger = logger
        self.index = faiss.IndexFlatIP(target_dim)  # 内积=余弦相似度（需L2归一化）
        self.pca = PCA(n_components=target_dim)
        self.is_pca_fitted = False

    def _normalize(self, v):
        """向量归一化处理"""
        norm = np.linalg.norm(v)
        if norm == 0:
            raise ValueError("Cannot normalize zero vector")
        return v / norm

    def add_vector(self, vector, thre=VECTOR_SEARCH_CONFIG['THRESHOLD']):
        """添加向量到索引"""
        try:
            vector = np.array(vector).astype('float32').reshape(1, -1)
            
            # # 如果PCA模型未训练，先用第一个向量训练
            # if not self.is_pca_fitted:
            #     self.pca.fit(vector)
            
            # 使用PCA进行降维
            # reduced_vector = self.pca.transform(vector)
            norm_v = self._normalize(vector.flatten())
        except ValueError as e:
            self.logger.error(f"[add_vector] 归一化失败: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"[add_vector] 向量处理异常: {str(e)}")
            return False

        # 空索引直接添加
        if self.index.ntotal == 0:
            self.index.add(norm_v)
            return True

        # 相似度检索
        try:
            similar, indices = self.index.search(norm_v, 1)
        except Exception as e:
            self.logger.error(f"[add_vector] 相似度检索失败: {str(e)}")
            return False

        if similar[0][0] > thre:
            self.logger.warning(
                f"[add_vector] 存在相似向量，跳过添加：相似度{similar[0][0]:.4f}，索引{indices[0][0]}。")
            return False

        self.index.add(norm_v)
        return True

    def import_from_mysql(self, hours_limit=None):
        """从MySQL导入向量数据"""
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # 先获取最近48小时的数据用于训练PCA
            # train_query = """
            #     SELECT feature 
            #     FROM news_detail 
            #     WHERE feature IS NOT NULL
            #     AND create_time >= DATE_SUB(NOW(), INTERVAL 48 HOUR)
            # """
            # cursor.execute(train_query)
            # train_results = cursor.fetchall()
            
            # if train_results and not self.is_pca_fitted:
            #     self.logger.info(f"开始使用最近48小时的数据训练PCA模型")
            #     # 准备训练数据
            #     train_features = []
            #     for (feature_blob,) in train_results:
            #         try:
            #             feature = np.frombuffer(feature_blob, dtype=np.float64)
            #             if len(feature) == self.dim:
            #                 train_features.append(feature)
            #         except Exception as e:
            #             self.logger.warning(f"处理训练数据时出错: {str(e)}")
            #             continue
            #     if train_features:
            #         # 使用最近48小时的数据训练PCA
            #         train_features = np.array(train_features)
            #         self.pca.fit(train_features)
            #         self.is_pca_fitted = True
            #         self.logger.info(f"使用{len(train_features)}条最新数据训练PCA模型完成")

            # 获取需要导入的向量数据
            base_query = """
                SELECT unique_id, feature 
                FROM news_detail 
                WHERE feature IS NOT NULL
            """
            
            if hours_limit is not None:
                query = base_query + " AND create_time >= DATE_SUB(NOW(), INTERVAL %s HOUR)"
                params = (hours_limit,)
            else:
                query = base_query
                params = ()

            cursor.execute(query, params)
            results = cursor.fetchall()
            total = len(results)
            success = 0

            for idx, (news_id, feature_blob) in enumerate(results, 1):
                try:
                    feature = np.frombuffer(feature_blob, dtype=np.float64)
                    if len(feature) != self.dim:
                        self.logger.warning(
                            f"新闻ID {news_id} 向量维度错误: 预期{self.dim}，实际{len(feature)}")
                        continue

                    if self.add_vector(feature):
                        success += 1

                except Exception as e:
                    self.logger.error(f"处理新闻ID {news_id} 失败: {str(e)}")
                    continue

                # 进度日志
                if idx % 100 == 0 or idx == total:
                    self.logger.info(f"处理进度: {idx}/{total} ({idx/total:.1%})")

            self.logger.info(f"导入完成: 成功{success}/{total} ({success/total:.1%})")

        except Exception as e:
            self.logger.error(f"数据库操作失败: {str(e)}")
            traceback.print_exc()
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals() and conn.is_connected():
                conn.close()

    def delete_all_embeddings(self):
        """清空所有向量数据"""
        self.index.reset()

    def display_embedding_nums(self):
        """返回索引中的向量数量"""
        return self.index.ntotal
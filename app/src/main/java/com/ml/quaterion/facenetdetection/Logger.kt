/*
 * Copyright 2023 Shubham Panchal
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.ml.quaterion.facenetdetection

// Logs message using log_textview present in activity_main.xml
class Logger {

    companion object {

        private const val MAX_LOG_LINES = 50

        fun log( message : String ) {
            val tv = MainActivity.logTextView
            tv.post {
                val lines = (tv.text.toString() + "\n>> $message").lines()
                val trimmed = if (lines.size > MAX_LOG_LINES) {
                    lines.takeLast(MAX_LOG_LINES).joinToString("\n")
                } else {
                    lines.joinToString("\n")
                }
                MainActivity.setMessage( trimmed )
                while ( tv.canScrollVertically(1) ) {
                    tv.scrollBy(0, 10)
                }
            }
        }

    }

}